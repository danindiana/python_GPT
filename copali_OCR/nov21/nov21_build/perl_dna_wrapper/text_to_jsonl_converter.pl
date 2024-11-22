#!/usr/bin/perl
use strict;
use warnings;
use JSON;
use File::Find;
use File::Basename;
use Term::ANSIColor;
use IO::Handle;

# Constants
my $JSONL_MAX_SIZE = 2 * 1024 * 1024 * 1024; # 2GB limit
my $output_dir = "./output";
my $jsonl_file_base = "enrolled_data";

# Variables
my $total_files_enrolled = 0;
my $running = 1;

# Function to log messages to console
sub log_message {
    my ($message) = @_;
    print color('cyan') . "[INFO]: " . color('reset') . $message . "\n";
}

# Function to prompt user for input
sub prompt {
    my ($message) = @_;
    print color('yellow') . $message . color('reset') . " ";
    my $response = <STDIN>;
    chomp($response);
    return $response;
}

# Function to process text files in a directory
sub process_directory {
    my ($target_dir, $recursive) = @_;
    my @files;
    my $file_counter = 0;
    log_message("Scanning directory: $target_dir (Recursive: $recursive)");

    # Use File::Find for recursive or non-recursive file collection
    if ($recursive) {
        find(sub { push @files, $File::Find::name if -f && /\.txt$/ }, $target_dir);
    } else {
        opendir(my $dir, $target_dir) or die "[ERROR]: Cannot open directory $target_dir: $!";
        @files = map { "$target_dir/$_" } grep { /\.txt$/ && -f "$target_dir/$_" } readdir($dir);
        closedir($dir);
    }

    my $current_jsonl_size = 0;
    my $jsonl_index = 1;
    my $current_jsonl_file = "$output_dir/${jsonl_file_base}_$jsonl_index.jsonl";

    open my $jsonl_fh, ">>", $current_jsonl_file or die "[ERROR]: Cannot open $current_jsonl_file: $!";
    foreach my $file (@files) {
        last unless $running; # Check for shutdown signal

        # Read text file content
        open my $fh, '<', $file or warn "[ERROR]: Cannot open $file: $!";
        my $content = do { local $/; <$fh> };
        close $fh;

        # Create JSON metadata
        my $metadata = {
            filename    => basename($file),
            path        => $file,
            size        => -s $file,
            modified_at => (stat($file))[9],
        };

        # Create JSON object
        my $json_object = {
            metadata => $metadata,
            content  => $content,
        };

        # Write to JSONL file
        my $json_text = encode_json($json_object) . "\n";
        print $jsonl_fh $json_text;
        $current_jsonl_size += length($json_text);

        # Check if current JSONL exceeds size limit
        if ($current_jsonl_size >= $JSONL_MAX_SIZE) {
            close $jsonl_fh;
            $jsonl_index++;
            $current_jsonl_file = "$output_dir/${jsonl_file_base}_$jsonl_index.jsonl";
            open $jsonl_fh, ">>", $current_jsonl_file or die "[ERROR]: Cannot open $current_jsonl_file: $!";
            $current_jsonl_size = 0;
        }

        log_message("Processed file: $file");
        $file_counter++;
    }
    close $jsonl_fh;

    $total_files_enrolled += $file_counter;
    log_message("Completed processing $file_counter files.");
}

# Function for graceful shutdown
sub shutdown_handler {
    log_message("Shutdown signal received. Stopping execution...");
    $running = 0;
}

# Setup signal handlers
$SIG{INT} = \&shutdown_handler;
$SIG{TERM} = \&shutdown_handler;

# Main Program Loop
while ($running) {
    my $target_dir = prompt("Enter the target directory to scan:");
    last unless -d $target_dir; # Exit if invalid directory

    my $recursive = prompt("Include files in subdirectories? (Y/N):");
    $recursive = ($recursive =~ /^(y|Y)$/) ? 1 : 0;

    # Ensure output directory exists
    mkdir $output_dir unless -d $output_dir;

    process_directory($target_dir, $recursive);

    log_message("Total files enrolled: $total_files_enrolled");
    my $continue = prompt("Do you want to process another directory? (Y/N):");
    last unless $continue =~ /^(y|Y)$/;
}

log_message("Program exited. Total files enrolled: $total_files_enrolled");
