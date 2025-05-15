# Scope

This repository provides access to the dataset as gathered in a systematic
mapping study on evaluation practices in the real-time systems research community
between 2017 and 2024. A paper presenting the process and analyzing the results
will be presented at ECRTS 2025 in Brussels, Belgium.

Further, the repository contains the python script used to conduct the data
analysis and create visualizations.

# Content
- *data.csv*
- *processData.py*
- *LICENSE-MIT*
- *LICENSE-CC-BY*
- *README.md (this document)*

# Tested platforms
The scripts were tested on the workstations by the two authors of the paper,
both running Ubuntu Linux 22.04 LTS.

The following python requirements must be fulfilled:
- `python3.10`
- `python3-pandas`
- `python3-numpy`
- `python3-matplotlib`

Optional requirements are:
- a PDF viewer
- a spreadsheet software
- a way to run an interactive python interpreter instance

# License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

The dataset is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

The python script is licensed under the terms of the MIT license.

# Running the script

To perform the full analysis once and to generate all the visuals, one can run
the script to completion by calling:

```
$ python3 processData.py
```

However, the interactive python3 interpreter gives additional possibilities in
perusing the dataset. This can be achieved by running:

```
$ python3
> exec(open("processData.py").read())
```

To get started analyzing the dataset, we recommend to follow the description of
the data gathering and categorization process as laid out in the main publication
and to look at the output of the following function calls in the python
interpreter:

```
> allTypes(df)
> allCategories(df)
> allSpecifier(df)
> allQualifier(df)
> allFields(df)
> allAuthors(df)
```

# References

The paper will be available to the public in the near future.
