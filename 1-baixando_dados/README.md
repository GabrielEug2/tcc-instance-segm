## Baixando dados

Eu usei dados do Openimages, com algumas das classes que mais aparecem no COCO, mas você pode usar qualquer dataset. Só salve as annotations no formato do COCO.

Para baixar um conjunto de teste como o que eu descrevi acima:
```bash
pip install fiftyone
python download_test_set.py <n_imgs> <out_dir>
```