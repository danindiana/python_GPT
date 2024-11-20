def log_error(file_name, error_message):
    with open("error.log", "a") as error_log:
        error_log.write(f"{file_name}: {error_message}\n")
