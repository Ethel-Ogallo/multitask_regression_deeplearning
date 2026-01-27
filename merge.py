import nbformat

notebooks = ["Multi-task_regression_deeplearning\\multitask-regression_final.ipynb",
             "Multi-task_regression_deeplearning\\multitask-regression_v1.ipynb"
             ]

merged = nbformat.v4.new_notebook(cells=[])

for nb in notebooks:
    with open(nb, "r", encoding="utf-8") as f:
        nbf = nbformat.read(f, as_version=4)
        merged.cells.extend(nbf.cells)

with open("Multi-task_regression_deeplearning\\multitask-regression_final_v2.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(merged, f)

# print("Merged notebook saved as Multi-task_regression_deeplearning\\multitask-regression_final.ipynb")