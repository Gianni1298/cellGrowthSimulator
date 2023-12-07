def parse_file_and_find_min_variance(filename):
    with open(filename, 'r') as file:
        min_variance = float('inf')
        min_variance_row = ""

        for line in file:
            if "Variance:" in line:
                # Extract the variance value
                variance_value = float(line.split("Variance:")[1].strip())
                # Update the minimum variance and corresponding row
                if variance_value < min_variance:
                    min_variance = variance_value
                    min_variance_row = line.strip()

        return min_variance_row


# Example usage (the file path needs to be correct for actual use)
min_variance_row = parse_file_and_find_min_variance("parameter_optimization_log.txt")
print(min_variance_row)
