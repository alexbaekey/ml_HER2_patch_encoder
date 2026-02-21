#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:-TCGA-BRCA}"
OUT_TSV="${2:-wsi_files.tsv}"

cat > query_wsi.json <<JSON
{
  "filters": {
    "op": "and",
    "content": [
      {"op":"=","content":{"field":"cases.project.project_id","value":"${PROJECT}"}},
      {"op":"=","content":{"field":"files.data_type","value":"Slide Image"}},
      {"op":"=","content":{"field":"files.data_format","value":"SVS"}},
      {"op":"=","content":{"field":"files.access","value":"open"}}
    ]
  },
  "format": "JSON",
  "fields": "file_id,file_name,file_size,data_format,data_type,access",
  "size": 20000
}
JSON

resp="$(curl -sS -X POST "https://api.gdc.cancer.gov/files" \
  -H "Content-Type: application/json" \
  -d @query_wsi.json)"

total="$(echo "$resp" | jq -r '.data.pagination.total')"
echo "Total hits: ${total}"

echo "$resp" | jq -r '.data.hits[] | [.file_id,.file_name,(.file_size|tostring)] | @tsv' > "$OUT_TSV"

echo "Found $(wc -l < "$OUT_TSV") WSI files (wrote $OUT_TSV)"
head -n 5 "$OUT_TSV" || true


awk -F '\t' '{sum+=$3} END {printf "Total size: %.2f GB\n", sum/1024/1024/1024}' wsi_files.tsv
