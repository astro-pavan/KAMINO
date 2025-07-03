
def modify_file_by_lines(filename: str, new_filename: str, modification_dict: dict[int, str]):

    try:
            # Read the file into a list of lines
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            # Apply modifications
            for line_number, new_content in modification_dict.items():
                if 1 <= line_number <= len(lines):
                    lines[line_number - 1] = new_content + '\n'
                else:
                    print(f"Line {line_number} is out of range. Skipping.")
            
            # Write the modified lines back to the file
            with open(new_filename, 'w') as file:
                file.writelines(lines)
        
    except FileNotFoundError:
        print(f"Error: The file at '{filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

