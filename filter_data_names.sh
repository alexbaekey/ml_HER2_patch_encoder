# filter to only diagnostic slides

grep -P "\t.*-DX1\." wsi_files.tsv > wsi_DX1.tsv
echo "DX1 files: $(wc -l < wsi_DX1.tsv)"
awk -F'\t' '{sum+=$3} END{printf "DX1 total: %.2f GB\n", sum/1024/1024/1024}' wsi_DX1.tsv
head -n 5 wsi_DX1.tsv

# filter to one slide per patient

awk -F'\t' '
{
  # filename is field 2, patient id is first 12 chars like TCGA-XX-YYYY
  pid = substr($2, 1, 12)
  if (!seen[pid]++) print
}' wsi_DX1.tsv > wsi_DX1_one_per_patient.tsv

echo "DX1 one-per-patient: $(wc -l < wsi_DX1_one_per_patient.tsv)"
awk -F'\t' '{sum+=$3} END{printf "Size: %.2f GB\n", sum/1024/1024/1024}' wsi_DX1_one_per_patient.tsv
