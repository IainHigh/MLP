import csv
import ijson

fields_to_keep = ["isbn", "language_code", "description", "isbn13", "book_id", "title", "num_pages"]
original_json_file = 'Datasets/goodreads_books.json'
cleaned_json_output = 'cleaned_json_books.json'
output_csv_file = 'goodreads_books.csv'

def clean_json(original_json_file):
    with open(original_json_file, 'r') as input_file:
        cleaned_json = [input_file.readline().rstrip() + ',' + '\n' for _ in input_file]
        
    cleaned_json[-2] = cleaned_json[-2][:-2]
    cleaned_json = cleaned_json[:-1]
        
    cleaned_json.insert(0, '[\n')
    cleaned_json.append('\n]')

    with open(cleaned_json_output, 'w') as output_file:
        output_file.writelines(cleaned_json)

    print("Json file cleaned", original_json_file)
    return cleaned_json



def convert_to_csv(fields_to_keep, output_csv_file):
    with open(cleaned_json_output, 'r') as file:
        parser = ijson.items(file, 'item')

        with open(output_csv_file, 'w', newline='') as csv_file:
            csvwriter = csv.DictWriter(csv_file, fieldnames=fields_to_keep)
            csvwriter.writeheader()
            for entry in parser:
                try:
                    entry['description'] = entry.get('description', '').replace('\n', ' ')
                    csvwriter.writerow({field: entry.get(field, '') for field in fields_to_keep})
                except Exception as e:
                    print(f"Error proecessing entry: {entry}. Error {e}")
                    continue

    print("Conversion complete. CSV file saved as", output_csv_file)

clean_json(original_json_file)
convert_to_csv(fields_to_keep, output_csv_file)

