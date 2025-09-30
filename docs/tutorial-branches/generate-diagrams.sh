#!/bin/bash

# This script generates PDF and PNG diagrams from the .tex files in each chapter directory.
#
# Dependencies:
# 1. A LaTeX distribution (e.g., MacTeX, MiKTeX, TeX Live) to use the 'pdflatex' command.
# 2. ImageMagick to use the 'magick' command for PDF to PNG conversion. On OSX, you can use brew: brew install imagemagick

#set -e # Exit immediately if a command exits with a non-zero status.

# Navigate to the script's directory to ensure correct relative paths
cd "$(dirname "$0")"

# Loop through each chapter directory
for chapter_dir in chapter-*/; do
    if [ -d "$chapter_dir" ]; then
        echo "Processing $chapter_dir..."
        cd "$chapter_dir"

        # Find the .tex file
        tex_file=$(find . -maxdepth 1 -name 'chapter-*.tex')
        echo "Found ${tex_file}"
        
        # For debugging
        #cd ..
        #continue 

        if [ -n "$tex_file" ]; then
            echo "  Found diagram file: $tex_file"
            
            # Generate the PDF
            echo "  Generating PDF..."
            pdflatex -interaction=nonstopmode "$tex_file"

            # Generate the PNG from the PDF
            pdf_file="${tex_file%.tex}.pdf"
            if [ -f "$pdf_file" ]; then
                echo "  Generating PNG from $pdf_file..."
                png_file="${tex_file%.tex}.png"
                magick -density 300 "$pdf_file" -quality 90 "$png_file"
                echo "  Successfully created $png_file"
            else
                echo "  Warning: PDF file not found, skipping PNG conversion."
            fi

            # Clean up auxiliary files
            echo "  Cleaning up auxiliary files..."
            rm -f "${tex_file%.tex}.aux" "${tex_file%.tex}.log"

        else
            echo "  No .tex file found in $chapter_dir"
        fi

        cd .. # Return to the parent directory
        echo "Done with $chapter_dir"
        echo "---------------------------------"
    fi
done

echo "All diagrams have been generated."
