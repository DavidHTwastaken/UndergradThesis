ENCODED_FILE="lrw.txt"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <URL> <OUTPUT_FILE>"
    exit 1
fi

URL="$1"
OUTPUT_FILE="$2"

# Decode Base64 and extract username/password directly into variables
CREDENTIALS=$(base64 -d "$ENCODED_FILE")
USERNAME=$(echo "$CREDENTIALS" | sed -n '1p')
PASSWORD=$(echo "$CREDENTIALS" | sed -n '2p')

# Use curl with the extracted credentials
curl -u "$USERNAME:$PASSWORD" -o "$OUTPUT_FILE" "$URL"