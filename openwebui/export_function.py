import json
import os.path

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "out")
FUNCTION_FILENAME = os.getenv("FUNCTION_FILENAME", "function.py")
OUTPUT_FUNCTION_JSON_FILENAME = os.getenv("OUTPUT_FUNCTION_JSON_FILENAME", "function-gadio.json")
VERSION = os.getenv("VERSION", "0.1.0")
                
def generate_function_json():
    with open(FUNCTION_FILENAME, 'r') as file:
        script_content = file.read()
    
    function_data = [{
        "id": "gadio",
        "name": "GADIO",
        "meta": {
            "description": "GADIO",
            "manifest": {
                "title": "GADIO Pipe",
                "author": "GADIO",
                "version": VERSION
            },
            "type": "pipe"
        },
        "content": script_content
    }]
    
    return function_data

if __name__ == "__main__":
    output_path = os.path.join(OUTPUT_PATH, OUTPUT_FUNCTION_JSON_FILENAME)
    function_data = generate_function_json()
    
    with open(output_path, 'w') as output_file:
        json.dump(function_data, output_file, indent=4)
    
    print(f"Exported Open WebUI compatible function JSON to {output_path}")