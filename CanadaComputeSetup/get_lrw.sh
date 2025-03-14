BASE_URL="https://thor.robots.ox.ac.uk/lip_reading/data1/lrw-v1-parta"
OUTPUT_FILE="lrw-v1.tar"

for PART in {a..g}; do
        FILE_NAME="lrw-v1-parta$PART"
        if [ -f "$FILE_NAME" ]; then
                echo "Skipping $FILE_NAME (already exists)"
        else
                ./download_part_lrw.sh "$BASE_URL$PART" $FILE_NAME
        fi
done

echo "Combining files into $OUTPUT_FILE..."
cat lrw-v1-parta{a..g} > "$OUTPUT_FILE"

# rm lrw-v1-parta{a..g}