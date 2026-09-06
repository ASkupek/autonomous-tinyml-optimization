## 🛑 Git Commit & Branching Strategy

To maintain a clean and structured repository history, follow the **Conventional Commits** standard for all git commits.

### Branching & Development Workflow
For any major feature, bug fix, or significant task, **always create a dedicated branch** before writing code:
- **Feature Branch:** `git checkout -b feature/your-feature-name`
- **Bug Fix Branch:** `git checkout -b fix/issue-description`

### Commit Format
`<type>: <short summary in lowercase>`

### Approved Types
- `feat:` A new feature or pipeline functionality (e.g., `feat: added early stopping to train pipeline`)
- `fix:` A bug fix (e.g., `fix: resolve port conflict between dashboard and kafka-ui`)
- `docs:` Documentation updates only (e.g., `docs: update devnotes with MCU porting guide`)
- `refactor:` Code refactoring without changing logic (e.g., `refactor: clean up argparse menu in main.py`)
- `test:` Adding or updating unit/integration tests (e.g., `test: add verification tests for int8 quantization`)

### Rules Before Pushing & Merging to Main
Before merging any feature or fix branch back into `main`, you must fulfill the following criteria:
1. **Write Tests:** Every new feature (`feat`) or bug fix (`fix`) must include corresponding unit or integration tests.
2. **Run Local Full-Flow Testing:** Perform a clean test cycle locally to ensure everything works end-to-end:
   ```bash
   # Tear down old infrastructure and volumes
   docker compose down -v
   
   # Start infrastructure
   docker compose up -d
   
   # Run the full test suite
   pytest -v
   
   # Verify CLI orchestrator execution
   python main.py --help