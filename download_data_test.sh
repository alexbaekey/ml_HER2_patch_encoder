head -n 5 wsi_DX1_one_per_patient.tsv > wsi_DX1_one_per_patient_5.tsv

mkdir -p TCGA_WSI_50
while IFS=$'\t' read -r file_id file_name file_size; do
  curl -L --fail --retry 10 --retry-delay 5 --connect-timeout 30 \
    -C - -o "TCGA_WSI_50/${file_name}" "https://api.gdc.cancer.gov/data/${file_id}"
done < wsi_DX1_one_per_patient_5.tsv
