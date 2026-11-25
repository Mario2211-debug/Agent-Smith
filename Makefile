# Agent Smith - install, run and test every component on its own.
# Run `make` (or `make help`) to list the targets.

UV        ?= uv
PY        ?= $(UV) run python
SANDBOX   ?= $(PY) -m src.sandbox.cli
TEST      := $(PY) -m unittest -v
CONFIG    ?= sandbox_template.json
PORT      ?= 8000
DEMO_REPO ?= /tmp/agent/demo_repo
MBPP_TASK ?= examples/mbpp_task.json
SWE_TASK  ?= examples/swebench_task.json
# MBPP agent (agent_mbpp.py) settings
MODEL     ?= openai/gpt-oss-120b
URL       ?= https://api.groq.com/openai/v1
ENV_FILE  ?= .env
CACHE     ?= cache

MCP_MBPP  := python mcp_tools_mbpp.py --task-file $(MBPP_TASK)
MCP_SWE   := python mcp_tools_swebench.py --local --root $(DEMO_REPO) --task-file $(SWE_TASK)

.DEFAULT_GOAL := help
.PHONY: help install lock test test-sandbox test-security test-tools \
	test-mcp test-extraction test-llm test-agent test-mbpp-agent test-one \
	check-cli sandbox sandbox-mbpp sandbox-swebench sandbox-http \
	manual-mbpp manual-swebench mcp-mbpp-http mcp-swebench-http \
	inspect-mbpp inspect-swebench demo demo-security demo-mbpp \
	demo-swebench demo-repo mbpp-dump mbpp-run mbpp-validate clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- setup
install: ## Install dependencies with uv
	$(UV) sync

lock: ## Refresh uv.lock after changing pyproject.toml
	$(UV) lock

# ------------------------------------------------------------ unit tests
test: ## Run every test
	$(PY) -m unittest discover -s tests -v

test-sandbox: ## Sandbox: security + behaviour (persistence, final_answer...)
	$(TEST) tests.test_sandbox_security tests.test_sandbox_behavior

test-security: ## Sandbox security only (imports, builtins, paths, network, timeout, memory)
	$(TEST) tests.test_sandbox_security

test-tools: ## Mandatory tools + MBPP run_tests, without MCP or agent
	$(TEST) tests.test_tools

test-mcp: ## Real MCP server over stdio wrapped by the sandbox
	$(TEST) tests.test_mcp_integration

test-extraction: ## Code extraction (python, XML, Hermes JSON, ReAct)
	$(TEST) tests.test_extraction

test-llm: ## Providers: key rotation, retries, fallback (no network)
	$(TEST) tests.test_llm_provider

test-agent: ## Agent loop with a scripted LLM and the real sandbox
	$(TEST) tests.test_agent_loop

test-mbpp-agent: ## Existing agent_mbpp.py tests
	$(TEST) tests.test_mbpp_request_metrics tests.test_mbpp_token_limits

test-one: ## Run one test: make test-one T=tests.test_tools.ToolTests
	$(TEST) $(T)

check-cli: ## Check that the `uv run sandbox` console script works
	$(UV) run sandbox -c "print('sandbox CLI ok')"

# --------------------------------------------------- interactive sandbox
sandbox: ## Interactive sandbox, no MCP server
	$(SANDBOX) $(CONFIG)

sandbox-mbpp: ## Interactive sandbox + MBPP MCP tools (stdio)
	$(SANDBOX) --mcp-stdio "$(MCP_MBPP)" $(CONFIG)

sandbox-swebench: demo-repo ## Interactive sandbox + SWE-bench tools on the demo repo
	$(SANDBOX) --mcp-stdio "$(MCP_SWE)" $(CONFIG)

sandbox-http: ## Sandbox connected to an HTTP MCP server (start mcp-*-http first)
	$(SANDBOX) --mcp-server http://127.0.0.1:$(PORT)/mcp $(CONFIG)

manual-mbpp: ## Print the LLM manual generated from the MBPP server
	$(SANDBOX) --mcp-stdio "$(MCP_MBPP)" --manual $(CONFIG)

manual-swebench: demo-repo ## Print the LLM manual generated from the SWE-bench server
	$(SANDBOX) --mcp-stdio "$(MCP_SWE)" --manual $(CONFIG)

# ------------------------------------------------------------ MCP servers
mcp-mbpp-http: ## Serve the MBPP tools over streamable HTTP on PORT
	$(PY) mcp_tools_mbpp.py --task-file $(MBPP_TASK) --http --port $(PORT)

mcp-swebench-http: demo-repo ## Serve the SWE-bench tools over HTTP on PORT
	$(PY) mcp_tools_swebench.py --local --root $(DEMO_REPO) \
		--task-file $(SWE_TASK) --http --port $(PORT)

inspect-mbpp: ## Open the MCP Inspector on the MBPP server
	$(UV) run mcp dev mcp_tools_mbpp.py

inspect-swebench: ## Open the MCP Inspector on the SWE-bench server
	$(UV) run mcp dev mcp_tools_swebench.py

# ------------------------------------------- demos (no agent, no LLM)
demo: demo-security demo-mbpp demo-swebench ## Run every demo

demo-security: ## Try each escape in examples/escape_attempts.txt
	@while IFS= read -r line; do \
		printf '\n\033[33m>>> %s\033[0m\n' "$$line"; \
		$(SANDBOX) examples/sandbox_fast.json -c "$$line"; \
	done < examples/escape_attempts.txt; true

demo-mbpp: ## Call the MBPP tools from the sandbox through MCP
	$(SANDBOX) --mcp-stdio "$(MCP_MBPP)" $(CONFIG) < examples/mbpp_demo.py

demo-swebench: demo-repo ## Call every mandatory tool from the sandbox through MCP
	$(SANDBOX) --mcp-stdio "$(MCP_SWE)" $(CONFIG) < examples/swebench_demo.py

demo-repo: ## (Re)create a small buggy git repo in DEMO_REPO
	@rm -rf $(DEMO_REPO) && mkdir -p $(DEMO_REPO)
	@printf 'def add(a, b):\n    return a - b\n\n\ndef total(values):\n    return sum(add(v, 0) for v in values)\n' > $(DEMO_REPO)/calc.py
	@printf 'import unittest\n\nfrom calc import add\n\n\nclass CalcTest(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n' > $(DEMO_REPO)/test_calc.py
	@cd $(DEMO_REPO) && git init -q && git add . && \
		git -c user.name=demo -c user.email=demo@example.com commit -qm init
	@echo "demo repo ready in $(DEMO_REPO)"

# ------------------------------------------------- MBPP agent + moulinette
mbpp-dump: ## Dump a random MBPP task to CACHE/mbpp_task.json
	@mkdir -p $(CACHE)
	cd moulinette && $(UV) run moulinette_eval dump mbpp --output ../$(CACHE)/mbpp_task.json

mbpp-run: ## Run agent_mbpp.py on CACHE/mbpp_task.json (MODEL, URL, ENV_FILE)
	$(UV) run --env-file $(ENV_FILE) python -m agent_mbpp \
		--task-file $(CACHE)/mbpp_task.json --output $(CACHE)/mbpp_solution.json \
		--model-name "$(MODEL)" --provider-url "$(URL)"

mbpp-validate: ## Validate CACHE/mbpp_solution.json with the moulinette
	cd moulinette && $(UV) run moulinette_eval validate mbpp \
		../$(CACHE)/mbpp_task.json ../$(CACHE)/mbpp_solution.json

# ------------------------------------------------------------------ misc
clean: ## Remove caches and the demo repo
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(DEMO_REPO)
