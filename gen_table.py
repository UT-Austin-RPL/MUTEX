from PIL import Image

def create_table(data):
    # Calculate number of rows and columns based on input data
    num_rows = len(data)
    num_cols = len(data[0])

    table = "<table>"

    # Loop through each row in the data
    for i in range(num_rows):
        table += "<tr>"

        # Loop through each element in the row
        for j in range(num_cols):
            cell_data = data[i][j]
            cell_tag = "td" if isinstance(cell_data, str) else "th"

            # Calculate cell width based on data type
            cell_width = "100px" if isinstance(cell_data, Image.Image) else "auto; white-space: pre-wrap"
            cell_style = f"style='width: {cell_width};'"

            # Add cell element to the table
            table += f"<{cell_tag} {cell_style}>"

            # Add cell data to the table
            if isinstance(cell_data, str) and cell_data.endswith(".png"):
              image_path = cell_data  # Assuming the image is in the same directory as the script

              table += f"<img src='{image_path}'/>"
            else:
                table += str(cell_data)

            table += f"</{cell_tag}>"

        table += "</tr>"

    table += "</table>"
    return table

# Example usage
data = [
    ["John", "Doe", "src/ditto.png"],
    ["Jane", "Smith", "src/ditto.png"],
]
html_table = create_table(data)
print(html_table)
