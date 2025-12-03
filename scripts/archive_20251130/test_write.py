import os
try:
    with open('scripts/test_output.txt', 'w') as f:
        f.write("Hello from Python!")
    print("File written successfully.")
except Exception as e:
    print(f"Error: {e}")
