# Checked example outputs

`all_lessons_seed42.json` captures the CLI's complete default-seed run in Python
3.12.14 with NumPy 2.3.5. It records the environment and each experiment's result.
Use it to learn the output structure and compare a first successful run. Treat
small floating-point differences as numerical variation, not automatic failure.

`book_programs/` contains stdout from each unchanged book listing. The Chapter 3
file is empty because its assertions are silent when successful. The original
programs use their own seeds; the new CLI's seed does not override them.

`verification.json` records the release checks. Numerical tests establish small
mathematical properties and execution behavior, not real-world model performance.
All data in these examples are synthetic. No inputs or outputs are downloaded.
