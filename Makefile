all:
	python3 ngnet_batch.py
install:
	python3 -m pip install -r requirements.txt
prof:
	python3 -m cProfile -o ngnet.prof -s tottime ngnet.py
