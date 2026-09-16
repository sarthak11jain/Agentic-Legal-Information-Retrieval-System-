# Reusable GitHub Project Blueprint

This project follows a repository pattern that can be reused for future portfolio projects. The goal is to make a repository understandable to a recruiter or collaborator within a few minutes while keeping the implementation reproducible for a technical reader.

## 1. Start with the project story

The root README should answer these questions in order:

1. What problem does the project solve?
2. What was built and why is the approach technically interesting?
3. What are the strongest verified results?
4. How does the system work?
5. How can someone run a safe, small example?
6. Where can someone find the detailed implementation and experiments?

Lead with the outcome and the system idea. Keep setup details below the first explanation of the project.

## 2. Use a predictable repository layout

```text
.
├── README.md                 # Project story, results, architecture, and quick start
├── src/                      # Stable, reusable implementation
├── scripts/                  # Human-facing commands and demos
├── tests/                    # Fast tests for important behavior
├── configs/                  # Non-secret configuration and documented defaults
├── examples/                 # Small, safe examples that run without private data
├── docs/                     # Architecture, reproduction, evaluation, and decisions
├── requirements.txt          # Pinned runtime dependencies
├── pyproject.toml            # Package/tooling metadata
├── .env.example              # Names of required environment variables, never values
├── .gitignore                # Data, secrets, caches, outputs, and environments
├── LICENSE                   # Explicit source-code license
└── CITATION.cff              # Optional citation metadata for research projects
```

Keep exploratory or historical code separate from the stable package. If the project is still evolving, document the boundary instead of presenting every experiment as production code.

## 3. Add a small end-to-end path

Every substantial project should have at least one command that demonstrates the core idea without requiring the full dataset, expensive model, private credentials, or external service. This gives a reviewer an immediate way to verify that the repository works.

The small path should be clearly labelled as a smoke demo or toy example. It must not be presented as a benchmark result.

## 4. Make claims traceable

Separate three types of results:

| Result type | How to present it |
| --- | --- |
| Official benchmark result | State the metric, split/submission, and source link |
| Experiment or ablation | State the configuration and point to the experiment log |
| Smoke-demo output | Label it illustrative and non-benchmark |

Avoid rounding or upgrading an experiment into an official result. A recruiter should be able to tell exactly which numbers support the CV.

## 5. Document the data boundary

State what is included, what is excluded, and how a reader can obtain or prepare required inputs. Never commit secrets, private data, hidden evaluation labels, credentials, or large generated artifacts merely to make the tree look complete.

If a dataset has competition, legal, privacy, or licensing restrictions, record the relevant boundary in `docs/data.md` and keep the public repository focused on code, schemas, and reproducible instructions.

## 6. Add lightweight quality gates

At minimum, include:

- fast unit tests for core transformations and scoring logic;
- a compile, lint, or type-check command appropriate to the language;
- a smoke demo;
- a CI workflow that runs the same checks on pushes and pull requests;
- a secret and large-file review before the first public push.

The checks should be fast enough that a future contributor will actually run them.

## 7. Use diagrams to explain relationships

Use one small architecture diagram for the main data or control flow. Add a second diagram only when it explains a different relationship, such as an evaluation loop, deployment path, or experiment design. Keep the diagram source editable and test its rendering on GitHub before publishing.

## 8. Treat configuration and credentials separately

Commit safe defaults and an `.env.example` file. Read real credentials from environment variables or a local secret manager. Do not place API keys, tokens, database passwords, private URLs, or credential-shaped placeholders in source files.

## 9. Finish the public-repository pass

Before sharing a repository, review:

- repository name, description, topics, and default branch;
- README title, result claims, links, diagrams, and code blocks;
- license and citation metadata;
- ignored files and repository size;
- test and CI status;
- accidental secrets, personal paths, private names, and generated outputs;
- whether a recruiter can understand the project without reading every source file.

This checklist is intentionally portable. For a future project, copy the structure and reasoning, then replace the domain-specific claims, architecture, metrics, data policy, and commands.
