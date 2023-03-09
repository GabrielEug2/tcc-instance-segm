
# As classes das predictions seguem a numeração do COCO (pessoa = 1, etc).
# As annotations baixadas do OpenImages usam uma numeração *diferente*
# (pessoa = 25?) então eu preciso mapear por nome. Eu tenho que fazer
# as duas "se entenderem" (pessoa ter o mesmo id nas duas)

# predictions into names, not ids
# annotations into names, not ids