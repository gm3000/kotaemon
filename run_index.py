import os
import sys
from gradio_client import Client, handle_file

# client = Client(
#   "http://127.0.0.1:7860/",
#   auth=["admin", "admin"]
# )

client = Client("http://127.0.0.1:7860/")

def run_index(file_paths):
    result = client.predict([handle_file(path) for path in file_paths], False, api_name="/index_fn_1")
    print(result)

def get_file_paths(folder_path):
    try:
        # Get all file paths in the specified folder
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f))]
        return file_paths
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied to access folder '{folder_path}'.")
        return []

def main():
    # Check if folder path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    
    # Get full file paths from the specified folder
    file_paths = get_file_paths(folder_path)
    
    # Print the list of full file paths
    if file_paths:
        print("Files in the folder:")
        for file_path in file_paths:
            print(file_path)
        run_index(file_paths)
    else:
        print("No files found in the folder.")

if __name__ == "__main__":
    main()
