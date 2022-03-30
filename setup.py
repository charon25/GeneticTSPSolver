import setuptools

with open("README.md", "r", encoding="utf-8") as fi:
    long_description = fi.read()

setuptools.setup(
	name="gentsp",
	version="0.0.1",
	author="Paul 'charon25' Kern",
	description="Travelling Salesman Problem solver using a genetic algorithm",
	long_description=long_description,
    long_description_content_type='text/markdown',
	python_requires=">=3.7",
	url="https://www.github.com/charon25/GenticTSPSolver",
	license="MIT",
	packages=['gentsp']
)