#!/usr/bin/perl
use strict;
use warnings;
use File::Slurp;
use File::Find;
use File::Basename;

# Directory where the output text files are stored
my $output_dir = '/home/bob/Downloads/'; # Replace with your directory path
my $big_file = "$output_dir/combined_output.txt";

# Open the big output file for writing
open(my $big_fh, '>', $big_file) or die "Could not open $big_file: $!";

# Process each file in the output directory
find(sub {
    # Only process .txt files
    return unless -f && /\.txt$/;

    my $file_path = $File::Find::name;
    my $content = read_file($file_path);

    # Check if file is empty
    if ($content =~ /^\s*$/) {
        unlink $file_path or warn "Could not delete empty file $file_path: $!";
    } else {
        # Append content to big output file
        print $big_fh $content, "\n";
    }
}, $output_dir);

close($big_fh);

print "All files have been processed, and combined output saved to $big_file\n";
