# Personal Notes

## O que fazer

Heavy-load (no Colab)

* Rodar os modelos nas imagens do COCO
	* Com quais splits?
		* Treino não, são muitas imagens e os modelos já viram elas.
	    * Validation... talvez. O problema é que eles podem ter sido usados no treinamento, então não faz muito sentido analisar elas.
		* Teste sim. O problema é que não tem annotations, então eu não posso calcular o IoU e AP. Mas considerando que esses resultados são os que tem no paper, não faz nem sentido calcular.
		* Unlabeled... talvez. Eu preciso ver melhor se essas imagens tem as mesmas classes do COCO.

* Segmentador pra imagens aleatórias
	* Poderia ser local, mas... instalar essas coisas é um saco. Deixa no Colab mesmo.
	* Bem simples: é só rodar os 3 modelos na imagem, salvar no formato adequado e plotar com a API do Detectron.

Plots e utils (local)

* Visualizar as imagens de teste e validation
	* Até poderia ser no Drive, mas... ia ter que ficar fazendo o upload das saídas toda hora. É melhor local mesmo.
	* Até teria como fazer sem as APIs, mas... nah, muito trabalho.

* IoU nas imagens de validação? Talvez, mais pra frente.

---

## Atualmente

* Rodar os modelos nas imagens de teste
	* Mask R-CNN pronto, falta YOLACT e SOLO

## Next

* Segmentator de imagens aleatórias
* Mais imagens de teste (1000)