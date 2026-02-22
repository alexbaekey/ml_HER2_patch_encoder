mkdir -p TCGA_WSI

while IFS=$'\t' read -r file_id file_name file_size; do
  out="TCGA_WSI/${file_name}"
  url="https://api.gdc.cancer.gov/data/${file_id}"

  echo "Downloading ${file_name} (${file_size} bytes)"
  curl -L --fail --retry 10 --retry-delay 5 --connect-timeout 30 \
    -C - -o "${out}" "${url}"
#done < wsi_files.tsv # all files
done < wsi_DX1_one_per_patient.tsv # filtered diagnostic, one slide per patient
