"""
MCP Server — HackerHouse TigerGraph Fraud Investigation
========================================================
Model Context Protocol server exposing fraud investigation tools
to LLM-based agents via a standard JSON-RPC 2.0 interface.
"""
import sys, os, json, logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.graph.engine import TigerGraphEngine
from backend.agent.fraud_agent import FraudInvestigationAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MCP] %(message)s")
logger = logging.getLogger("mcp_server")

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOOLS = {
    "card_window": {
        "name": "card_window",
        "description": "Returns all transactions for a card within a time window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string"},
                "target_ts": {"type": "string", "description": "ISO timestamp"},
                "hours_before": {"type": "integer", "default": 48},
                "hours_after": {"type": "integer", "default": 48}
            },
            "required": ["card_id", "target_ts"]
        }
    },
    "device_neighbors": {
        "name": "device_neighbors",
        "description": "Finds all transactions and linked fraud cases sharing a device profile.",
        "inputSchema": {
            "type": "object",
            "properties": {"device_profile": {"type": "string"}},
            "required": ["device_profile"]
        }
    },
    "card_testing_check": {
        "name": "card_testing_check",
        "description": "Detects card testing pattern: micro-auths followed by larger purchases.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string"},
                "target_ts": {"type": "string"}
            },
            "required": ["card_id", "target_ts"]
        }
    },
    "investigate_case": {
        "name": "investigate_case",
        "description": "Runs full fraud investigation on a benchmark case and returns verdict.",
        "inputSchema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"]
        }
    },
    "customer_baseline": {
        "name": "customer_baseline",
        "description": "Returns historical transaction baseline for a customer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "before_ts": {"type": "string"}
            },
            "required": ["customer_id", "before_ts"]
        }
    }
}


class MCPHandler(BaseHTTPRequestHandler):
    engine = None
    agent = None
    cases_cache = {}

    def log_message(self, format, *args):
        logger.info(format % args)

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/tools":
            self._json_response({"tools": list(TOOLS.values())})
        elif path == "/health":
            self._json_response({"status": "ok", "engine": "TigerGraph in-memory"})
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_POST(self):
        from datetime import datetime
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        path = urlparse(self.path).path

        if path != "/call":
            self._json_response({"error": "Only /call endpoint supported"}, 404)
            return

        tool_name = body.get("name")
        params = body.get("arguments", {})

        try:
            if tool_name == "card_window":
                from datetime import datetime as dt
                ts = dt.strptime(params["target_ts"], "%Y-%m-%d %H:%M:%S")
                result = MCPHandler.engine.query_card_window(
                    params["card_id"], ts,
                    params.get("hours_before", 48),
                    params.get("hours_after", 48)
                )
                self._json_response({"content": [{"type": "json", "data": result}]})

            elif tool_name == "device_neighbors":
                result = MCPHandler.engine.query_device_neighbors(params["device_profile"])
                self._json_response({"content": [{"type": "json", "data": result}]})

            elif tool_name == "card_testing_check":
                from datetime import datetime as dt
                ts = dt.strptime(params["target_ts"], "%Y-%m-%d %H:%M:%S")
                result = MCPHandler.engine.detect_card_testing(params["card_id"], ts)
                self._json_response({"content": [{"type": "json", "data": result}]})

            elif tool_name == "investigate_case":
                case_id = params["case_id"]
                cases_dir = os.path.join(DATA_DIR, "cases")
                case_file = os.path.join(cases_dir, f"{case_id}.json")
                if os.path.exists(case_file):
                    with open(case_file) as f:
                        result = json.load(f)
                else:
                    result = {"error": f"Case {case_id} not yet investigated. Run fraud_agent.py first."}
                self._json_response({"content": [{"type": "json", "data": result}]})

            elif tool_name == "customer_baseline":
                from datetime import datetime as dt
                ts = dt.strptime(params["before_ts"], "%Y-%m-%d %H:%M:%S")
                result = MCPHandler.engine.query_customer_baseline(params["customer_id"], ts)
                self._json_response({"content": [{"type": "json", "data": result}]})

            else:
                self._json_response({"error": f"Unknown tool: {tool_name}"}, 400)

        except Exception as e:
            logger.error(f"Tool call error: {e}")
            self._json_response({"error": str(e)}, 500)


def main(port=8080):
    logger.info(f"Loading engine from {DATA_DIR}...")
    MCPHandler.engine = TigerGraphEngine(DATA_DIR)
    logger.info(f"Starting MCP server on port {port}...")
    server = HTTPServer(("0.0.0.0", port), MCPHandler)
    logger.info(f"MCP server ready at http://localhost:{port}")
    logger.info(f"  GET  /tools   — list available tools")
    logger.info(f"  GET  /health  — server health")
    logger.info(f"  POST /call    — call a tool")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down MCP server.")


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    main(port)
