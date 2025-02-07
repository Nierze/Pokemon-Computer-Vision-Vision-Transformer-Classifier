import os
import sys
import nbconvert

def convert_notebooks_to_py(notebooks_dir):
    # Ensure the provided path exists
    if not os.path.isdir(notebooks_dir):
        print(f"Error: Directory '{notebooks_dir}' does not exist.")
        return
    
    # Create a new folder for converted files
    output_dir = os.path.join(os.path.dirname(notebooks_dir), "Python Version")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .ipynb files in the directory
    notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")]
    
    if not notebook_files:
        print("No .ipynb files found in the directory.")
        return
    
    # Convert each notebook to .py
    for notebook in notebook_files:
        notebook_path = os.path.join(notebooks_dir, notebook)
        output_file = os.path.join(output_dir, notebook.replace(".ipynb", ".py"))
        
        with open(output_file, "w", encoding="utf-8") as f:
            exporter = nbconvert.PythonExporter()
            body, _ = exporter.from_filename(notebook_path)
            f.write(body)
        
        print(f"Converted: {notebook} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_ipynb_to_py.py <absolute_path_to_notebooks_folder>")
    else:
        convert_notebooks_to_py(sys.argv[1])
