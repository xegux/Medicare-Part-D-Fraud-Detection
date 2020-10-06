import numpy as np
import h5py
import csv
import argparse


def main(input_file_names, output_file_name, key_column_name="npi", time_column_name="year", first_column_clean=True,
         numpy_dtype="integer32"):

    # First pass to determine the dimensions of the matrix that needs to be generated

    key_dict = {}
    time_list = []  # This will store time element offsets

    i = 0  # Counter for position in CSV file
    j = 0  # Counter for position in HDF5 file

    for input_file_name in input_file_names:
        print("Scanning '%s' file" % input_file_name)
        with open(input_file_name, newline="", mode="r") as f:
            csv_reader = csv.reader(f)

            header = csv_reader.__next__()

            if first_column_clean:
                header[0] = header[0][3:] # Sample CSV file had weird characters only for first column

            column_index_dict = {header[i]: i for i in range(len(header))}
            #reverse_index_dict = {i: header[i] for i in range(len(header))}

            for row in csv_reader:

                key_value = row[column_index_dict[key_column_name]]

                if key_value not in key_dict: # Store starting position associated with the key
                    key_dict[key_value] = (j, i)
                    j += 1
                i += 1

                time_value = row[column_index_dict[time_column_name]]

                if time_value not in time_list:
                    time_list += [time_value]

    time_list.sort()
    time_dict = {time_list[i]: i for i in range(len(time_list))}

    # First we create the HDF5 file that we are going to write to

    with h5py.File(output_file_name, "w") as f5w:

        # Second pass through the CSV file this time to write

        key_values_group = f5w.create_group("/key")  # Store key values which are NPIs of provider as string
        time_value_group = f5w.create_group("/time") # Store time values which are years as strings

        data_value_group = f5w.create_group("/data")

        number_of_keys = len(key_dict)
        number_of_time_values = len(time_dict)

        columns_to_ignore = [key_column_name, time_column_name]
        data_columns = [x for x in header if x not in columns_to_ignore]

        column_positions_to_ignore = [column_index_dict[x] for x in columns_to_ignore]

        number_of_data_columns = len(data_columns)

        print("Number of keys (items): %s" % number_of_keys)
        print("Number of data columns: %s" % number_of_data_columns)
        print("Number of time steps: %s" % number_of_time_values)
        print("")

        key_values_ds = key_values_group.create_dataset("values", shape=(number_of_keys,1), dtype="S32", compression="gzip")
        time_values_ds = time_value_group.create_dataset("values", shape=(number_of_time_values,1), dtype="S128")
        time_values_ds[...] = np.reshape(np.array(time_list, dtype="S128"), newshape=(number_of_time_values, 1))

        data_value_array_ds = data_value_group.create_dataset("core_array",
                                                              shape=(number_of_keys, number_of_time_values, number_of_data_columns),
                                                              dtype=numpy_dtype, compression=None)

        data_value_annotations_ds = data_value_group.create_dataset("column_labels", shape=(1, number_of_data_columns), dtype="S256")

        data_value_annotations_ds[...] = np.reshape(np.array(data_columns, dtype="S256"), newshape=(1, number_of_data_columns))

        processed_keys = {}
        key_data_array = np.zeros((number_of_time_values, number_of_data_columns), dtype=numpy_dtype)
        key_value_array = np.empty(number_of_keys, dtype="S255")

        for input_file_name in input_file_names:

            print("Processing %s file" % input_file_name)

            with open(input_file_name, newline="", mode="r") as f:

                csv_reader = csv.reader(f)
                csv_reader.__next__()  # Skip header

                i = 0
                past_position_key_value = 0
                past_key_value = None
                for row in csv_reader:

                    key_value = row[column_index_dict[key_column_name]]
                    position_key_value, _ = key_dict[key_value]

                    time_value = row[column_index_dict[time_column_name]]
                    time_position = time_dict[time_value]

                    if key_value not in processed_keys and position_key_value > 0:  # We hit a new key
                        # print(i, position_key_value, past_position_key_value)
                        data_value_array_ds[past_position_key_value, :, :] = key_data_array
                        key_data_array = np.zeros((number_of_time_values, number_of_data_columns), dtype=numpy_dtype) # Reset to zeros
                        key_value_array[past_position_key_value] = past_key_value  # Write numpy position

                        processed_keys[key_value] = 1

                    row_to_write = []
                    for j in range(len(row)):
                        if j not in column_positions_to_ignore:
                            row_to_write += [float(row[j])]

                    key_data_array[time_position, :] = np.array(row_to_write, dtype=numpy_dtype)

                    if i > 0 and i % 5000 == 0: # Print rows read
                        print("Read %s rows" % i)

                    i += 1

                    past_position_key_value = position_key_value
                    past_key_value = key_value

            # Make sure we don't miss the last data elements
            key_value_array[past_position_key_value] = past_key_value
            data_value_array_ds[past_position_key_value, :, :] = key_data_array

            key_values_ds[...] = np.reshape(key_value_array, newshape=(number_of_keys, 1))


if __name__ == "__main__":
    arg_parse_obj = argparse.ArgumentParser(description="Read in a CSV file and build an HDF5 file")
    #arg_parse_obj.add_argument("-f", "--input-file-names", default="./5year_npi_generic_state_small_sortbyNPI.csv", dest="input_file_names")
    arg_parse_obj.add_argument("-f", "--input-file-names", default="./npi_exclusion_generic_2017_us.csv ./npi_exclusion_generic_2016_us.csv ./npi_exclusion_generic_2015_us.csv ./npi_exclusion_generic_2014_us.csv ./npi_exclusion_generic_2013_us.csv", dest="input_file_names")
    arg_parse_obj.add_argument("-o", "--output-file-name", default="./prescriber_5year.hdf5", dest="output_file_name")
    arg_parse_obj.add_argument("-d", "--numpy-dtype", default="int32", dest="numpy_dtype")

    arg_obj = arg_parse_obj.parse_args()
    input_file_names = arg_obj.input_file_names.split()
    main(input_file_names, arg_obj.output_file_name, numpy_dtype=arg_obj.numpy_dtype)