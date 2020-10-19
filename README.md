# Awesome Machine Learning On Source Code [![Awesome Machine Learning On Source Code](badges/awesome.svg)](https://github.com/src-d/awesome-machine-learning-on-source-code) [![CI Status](https://travis-ci.org/src-d/awesome-machine-learning-on-source-code.svg)](https://travis-ci.org/src-d/awesome-machine-learning-on-source-code)

![Awesome Machine Learning On Source Code](img/awesome-machine-learning-artwork.png)

**Notice: This repository is no longer actively maintained, and no further updates will be done, nor issues/PRs will be answered or attended.**
An alternative actively maintained can be found at [ml4code.github.io](https://ml4code.github.io/papers.html) [repository](https://github.com/ml4code/ml4code.github.io).

A curated list of awesome research papers, datasets and software projects devoted to machine learning _and_ source code. [#MLonCode](https://twitter.com/hashtag/MLonCode)

## Contents

- [Digests](#digests)
- [Conferences](#conferences)
- [Competitions](#competitions)
- [Papers](#papers)
  - [Program Synthesis and Induction](#program-synthesis-and-induction)
  - [Source Code Analysis and Language modeling](#source-code-analysis-and-language-modeling)
  - [Neural Network Architectures and Algorithms](#neural-network-architectures-and-algorithms)
  - [Embeddings in Software Engineering](#embeddings-in-software-engineering)
  - [Program Translation](#program-translation)
  - [Code Suggestion and Completion](#code-suggestion-and-completion)
  - [Program Repair and Bug Detection](#program-repair-and-bug-detection)
  - [APIs and Code Mining](#apis-and-code-mining)
  - [Code Optimization](#code-optimization)
  - [Topic Modeling](#topic-modeling)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Code Summarization](#code-summarization)
  - [Clone Detection](#clone-detection)
  - [Differentiable Interpreters](#differentiable-interpreters)
  - [Related research](#related-research)<details><summary>(links require "Related research" spoiler to be open)</summary>
    - [AST Differencing](#ast-differencing)
    - [Binary Data Modeling](#binary-data-modeling)
    - [Soft Clustering Using T-mixture Models](#soft-clustering-using-t-mixture-models)
    - [Natural Language Parsing and Comprehension](#natural-language-parsing-and-comprehension)
      </details>
- [Posts](#posts)
- [Talks](#talks)
- [Software](#software)
  - [Machine Learning](#machine-learning)
  - [Utilities](#utilities)
- [Datasets](#datasets)
- [Credits](#credits)
- [Contributions](#contributions)
- [License](#license)

## Digests

- [Learning from "Big Code"](http://learnbigcode.github.io) - Techniques, challenges, tools, datasets on "Big Code".
- [A Survey of Machine Learning for Big Code and Naturalness](https://ml4code.github.io/) - Survey and literature review on Machine Learning on Source Code.

## Conferences

- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [ACM International Conference on Software Engineering, ICSE](https://www.icse2018.org/)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [ACM International Conference on Automated Software Engineering, ASE](https://2019.aseconf.org)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (FSE)](https://conf.researchr.org/home/fse-2018)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [2018 IEEE 25th International Conference on Software Analysis, Evolution, and Reengineering (SANER)](https://www.conference-publishing.com/list.php?Event=SANER18MAIN)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [Machine Learning for Programming](https://ml4p.org/)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [Workshop on NLP for Software Engineering](https://nl4se.github.io/)
- <img src="badges/origin-industry-green.svg" alt="origin-industry" align="top"> [SysML](http://www.sysml.cc/)
  - [Talks](https://www.youtube.com/channel/UChutDKIa-AYyAmbT45s991g/)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [Mining Software Repositories](http://www.msrconf.org/)
- <img src="badges/origin-industry-green.svg" alt="origin-industry" align="top"> [AIFORSE](https://aiforse.org/)
- <img src="badges/origin-industry-green.svg" alt="origin-industry" align="top"> [source{d} tech talks](https://blog.sourced.tech/post/ml_talks_moscow/)
  - [Talks](https://www.youtube.com/playlist?list=PL5Ld68ole7j3iQFUSB3fR9122dHCUWXsy)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [NIPS Neural Abstract Machines and Program Induction workshop](https://uclmr.github.io/nampi/)
  - [Talks](https://www.youtube.com/playlist?list=PLzTDea_cM27LVPSTdK9RypSyqBHZWPywt)
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [CamAIML](https://www.microsoft.com/en-us/research/event/artificial-intelligence-and-machine-learning-in-cambridge-2017/)
  - [Learning to Code: Machine Learning for Program Induction](https://www.youtube.com/watch?v=vzDuVhFMB9Q) - Alexander Gaunt.
- <img src="badges/origin-academia-blue.svg" alt="origin-academia" align="top"> [MASES 2018](https://mases18.github.io/)

## Competitions

- [CodRep](https://github.com/KTH/CodRep-competition) - competition on automatic program repair: given a source line, find the insertion point.

## Papers

#### Program Synthesis and Induction

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Program Synthesis and Semantic Parsing with Learned Code Idioms](https://arxiv.org/abs/1906.10816v2) - Richard Shin, Miltiadis Allamanis, Marc Brockschmidt, Oleksandr Polozov, 2019.
- <img src="badges/16-pages-gray.svg" alt="16-pages" align="top"> [Synthetic Datasets for Neural Program Synthesis](https://openreview.net/forum?id=ryeOSnAqYm) - Richard Shin, Neel Kant, Kavi Gupta, Chris Bender, Brandon Trabucco, Rishabh Singh, Dawn Song, ICLR 2019.
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [Execution-Guided Neural Program Synthesis](https://openreview.net/forum?id=H1gfOiAqYm) - Xinyun Chen, Chang Liu, Dawn Song, ICLR 2019.
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [DeepFuzz: Automatic Generation of Syntax Valid C Programs for Fuzz Testing](https://faculty.ist.psu.edu/wu/papers/DeepFuzz.pdf) - Xiao Liu, Xiaoting Li, Rupesh Prajapati, Dinghao Wu, AAAI 2019.
- <img src="badges/12-pages-beginner-brightgreen.svg" alt="12-pages-beginner" align="top"> [NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System](https://arxiv.org/abs/1802.08979v2) - Xi Victoria Lin, Chenglong Wang, Luke Zettlemoyer, Michael D. Ernst, LREC 2018.
- <img src="badges/18-pages-gray.svg" alt="18-pages" align="top"> [Recent Advances in Neural Program Synthesis](https://arxiv.org/abs/1802.02353v1) - Neel Kant, 2018.
- <img src="badges/16-pages-gray.svg" alt="16-pages" align="top"> [Neural Sketch Learning for Conditional Program Generation](https://arxiv.org/abs/1703.05698) - Vijayaraghavan Murali, Letao Qi, Swarat Chaudhuri, Chris Jermaine, ICLR 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Neural Program Search: Solving Programming Tasks from Description and Examples](https://arxiv.org/abs/1802.04335v1) - Illia Polosukhin, Alexander Skidanov, ICLR 2018.
- <img src="badges/16-pages-gray.svg" alt="16-pages" align="top"> [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526v1) - Daniel A. Abolafia, Mohammad Norouzi, Quoc V. Le, 2018.
- <img src="badges/31-pages-gray.svg" alt="31-pages" align="top"> [Towards Synthesizing Complex Programs from Input-Output Examples](https://arxiv.org/abs/1706.01284v3) - Xinyun Chen, Chang Liu, Dawn Song, ICLR 2018.
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [Glass-Box Program Synthesis: A Machine Learning Approach](https://arxiv.org/abs/1709.08669v1) - Konstantina Christakopoulou, Adam Tauman Kalai, AAAI 2018.
- <img src="badges/14-pages-beginner-brightgreen.svg" alt="14-pages" align="top"> [Synthesizing Benchmarks for Predictive Modeling](https://chriscummins.cc/pub/2017-cgo.pdf) - Chris Cummins, Pavlos Petoumenos, Zheng Wang, Hugh Leather, CGO 2017
- <img src="badges/17-pages-beginner-brightgreen.svg" alt="17-pages-beginner" align="top"> [Program Synthesis for Character Level Language Modeling](https://files.sri.inf.ethz.ch/website/papers/charmodel-iclr2017.pdf) - Pavol Bielik, Veselin Raychev, Martin Vechev, ICLR 2017.
- <img src="badges/13-pages-beginner-brightgreen.svg" alt="13-pages-beginner" align="top"> [SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning](https://arxiv.org/abs/1711.04436v1) - Xiaojun Xu, Chang Liu, Dawn Song, 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Learning to Select Examples for Program Synthesis](https://arxiv.org/abs/1711.03243v1) - Yewen Pu, Zachery Miranda, Armando Solar-Lezama, Leslie Pack Kaelbling, 2017.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Neural Program Meta-Induction](https://arxiv.org/abs/1710.04157v1) - Jacob Devlin, Rudy Bunel, Rishabh Singh, Matthew Hausknecht, Pushmeet Kohli, NIPS 2017.
- <img src="badges/14-pages-beginner-brightgreen.svg" alt="14-pages-beginner" align="top"> [Learning to Infer Graphics Programs from Hand-Drawn Images](https://arxiv.org/abs/1707.09627v4) - Kevin Ellis, Daniel Ritchie, Armando Solar-Lezama, Joshua B. Tenenbaum, 2017.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Neural Attribute Machines for Program Generation](https://arxiv.org/abs/1705.09231v2) - Matthew Amodio, Swarat Chaudhuri, Thomas Reps, 2017.
- <img src="badges/11-pages-beginner-brightgreen.svg" alt="11-pages-beginner" align="top"> [Abstract Syntax Networks for Code Generation and Semantic Parsing](https://arxiv.org/abs/1704.07535v1) - Maxim Rabinovich, Mitchell Stern, Dan Klein, ACL 2017.
- <img src="badges/20-pages-gray.svg" alt="20-pages" align="top"> [Making Neural Programming Architectures Generalize via Recursion](https://arxiv.org/pdf/1704.06611v1.pdf) - Jonathon Cai, Richard Shin, Dawn Song, ICLR 2017.
- <img src="badges/14-pages-gray.svg" alt="14-pages" align="top"> [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696v1) - Pengcheng Yin, Graham Neubig, ACL 2017.
- <img src="badges/12-pages-beginner-brightgreen.svg" alt="12-pages-beginner" align="top"> [Program Synthesis from Natural Language Using Recurrent Neural Networks](https://homes.cs.washington.edu/~mernst/pubs/nl-command-tr170301.pdf) - Xi Victoria Lin, Chenglong Wang, Deric Pang, Kevin Vu, Luke Zettlemoyer, Michael Ernst, 2017.
- <img src="badges/18-pages-beginner-brightgreen.svg" alt="18-pages-beginner" align="top"> [RobustFill: Neural Program Learning under Noisy I/O](https://arxiv.org/abs/1703.07469v1) - Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, Pushmeet Kohli, ICML 2017.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Lifelong Perceptual Programming By Example](https://openreview.net/pdf?id=HJStZKqel) - Gaunt, Alexander L., Marc Brockschmidt, Nate Kushman, and Daniel Tarlow, 2017.
- <img src="badges/7-pages-gray.svg" alt="7-pages" align="top"> [Neural Programming by Example](https://arxiv.org/abs/1703.04990v1) - Chengxun Shu, Hongyu Zhang, AAAI 2017.
- <img src="badges/21-pages-gray.svg" alt="21-pages" align="top"> [DeepCoder: Learning to Write Programs](https://arxiv.org/abs/1611.01989) - Balog Matej, Alexander L. Gaunt, Marc Brockschmidt, Sebastian Nowozin, and Daniel Tarlow, ICLR 2017.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [A Differentiable Approach to Inductive Logic Programming](https://pdfs.semanticscholar.org/9698/409fc1603d28b6d51c38261f6243837c8bdd.pdf) - Yang Fan, Zhilin Yang, and William W. Cohen, 2017.
- <img src="badges/12-pages-beginner-brightgreen.svg" alt="12-pages-beginner" align="top"> [Latent Attention For If-Then Program Synthesis](https://arxiv.org/abs/1611.01867v1) - Xinyun Chen, Chang Liu, Richard Shin, Dawn Song, Mingcheng Chen, NIPS 2016.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top" id="card2code"> [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744) - Wang Ling, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočiský, Andrew Senior, Fumin Wang, Phil Blunsom, ACL 2016.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision (Short Version)](https://arxiv.org/abs/1612.01197) - Liang Chen, Jonathan Berant, Quoc Le, Kenneth D. Forbus, and Ni Lao, NIPS 2016.
- <img src="badges/5-pages-gray.svg" alt="5-pages" align="top"> [Programs as Black-Box Explanations](https://arxiv.org/abs/1611.07579) - Singh, Sameer, Marco Tulio Ribeiro, and Carlos Guestrin, NIPS 2016.
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [Search-Based Generalization and Refinement of Code Templates](http://soft.vub.ac.be/Publications/2016/vub-soft-tr-16-06.pdf) - Tim Molderez, Coen De Roover, SSBSE 2016.
- <img src="badges/14-pages-gray.svg" alt="14-pages" align="top"> [Structured Generative Models of Natural Source Code](https://arxiv.org/abs/1401.0514) - Chris J. Maddison, Daniel Tarlow, ICML 2014.

#### Source Code Analysis and Language modeling

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Modeling Vocabulary for Big Code Machine Learning](https://arxiv.org/abs/1904.01873v1) - Hlib Babii, Andrea Janes, Romain Robbes, 2019.
- <img src="badges/24-pages-gray.svg" alt="24-pages" align="top"> [Generative Code Modeling with Graphs](https://openreview.net/forum?id=Bke4KsA5FX) - Marc Brockschmidt, Miltiadis Allamanis, Alexander L. Gaunt, Oleksandr Polozov, ICLR 2019.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [NL2Type: Inferring JavaScript Function Types from Natural Language Information](http://software-lab.org/publications/icse2019_NL2Type.pdf) - Rabee Sohail Malik, Jibesh Patra, Michael Pradel, ICSE 2019.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [A Novel Neural Source Code Representation based on Abstract Syntax Tree](http://xuwang.tech/paper/astnn_icse2019.pdf) - Jian Zhang, Xu Wang, Hongyu Zhang, Hailong Sun, Kaixuan Wang, Xudong Liu, ICSE 2019.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Deep Learning Type Inference](http://vhellendoorn.github.io/PDF/fse2018-j2t.pdf) - Vincent J. Hellendoorn, Christian Bird, Earl T. Barr and Miltiadis Allamanis, FSE 2018. [Code](https://github.com/DeepTyper/DeepTyper).
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Tree2Tree Neural Translation Model for Learning Source Code Changes](https://arxiv.org/pdf/1810.00314.pdf) - Saikat Chakraborty, Miltiadis Allamanis, Baishakhi Ray, 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [code2seq: Generating Sequences from Structured Representations of Code](https://arxiv.org/abs/1808.01400) - Uri Alon, Omer Levy, Eran Yahav, 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Syntax and Sensibility: Using language models to detect and correct syntax errors](http://softwareprocess.es/pubs/santos2018SANER-syntax.pdf) - Eddie Antonio Santos, Joshua Charles Campbell, Dhvani Patel, Abram Hindle, and José Nelson Amaral, SANER 2018.
- <img src="badges/25-pages-gray.svg" alt="25-pages" align="top"> [code2vec: Learning Distributed Representations of Code](https://arxiv.org/abs/1803.09473v2) - Uri Alon, Meital Zilberstein, Omer Levy, Eran Yahav, 2018.
- <img src="badges/16-pages-gray.svg" alt="16-pages" align="top"> [Learning to Represent Programs with Graphs](https://arxiv.org/abs/1711.00740v1) - Miltiadis Allamanis, Marc Brockschmidt, Mahmoud Khademi, ICLR 2018.
- <img src="badges/36-pages-gray.svg" alt="36-pages" align="top"> [A Survey of Machine Learning for Big Code and Naturalness](https://arxiv.org/abs/1709.06182v1) - Miltiadis Allamanis, Earl T. Barr, Premkumar Devanbu, Charles Sutton, 2017.
- <img src="badges/36-pages-gray.svg" alt="36-pages" align="top"> [Are Deep Neural Networks the Best Choice for Modeling Source Code?](http://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf) - Vincent J. Hellendoorn, Premkumar Devanbu, FSE 2017.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [A deep language model for software code](https://arxiv.org/abs/1608.02715v1) - Hoa Khanh Dam, Truyen Tran, Trang Pham, 2016.
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [Convolutional Neural Networks over Tree Structures for Programming Language Processing](https://arxiv.org/abs/1409.5718) - Lili Mou, Ge Li, Lu Zhang, Tao Wang, Zhi Jin, AAAI-16. [Code](https://github.com/crestonbunch/tbcnn).
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Suggesting Accurate Method and Class Names](http://homepages.inf.ed.ac.uk/csutton/publications/accurate-method-and-class.pdf) - Miltiadis Allamanis, Earl T. Barr, Christian Bird, Charles Sutton, FSE 2015.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Mining Source Code Repositories at Massive Scale using Language Modeling](http://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf) - Miltiadis Allamanis, Charles Sutton, MSR 2013.

#### Neural Network Architectures and Algorithms

- <img src="badges/19-pages-gray.svg" alt="19-pages" align="top"> [Learning Compositional Neural Programs with Recursive Tree Search and Planning](https://arxiv.org/abs/1905.12941v1) - Thomas Pierrot, Guillaume Ligner, Scott Reed, Olivier Sigaud, Nicolas Perrin, Alexandre Laterre, David Kas, Karim Beguir, Nando de Freitas, 2019.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [From Programs to Interpretable Deep Models and Back](https://link.springer.com/content/pdf/10.1007%2F978-3-319-96145-3_2.pdf) - Eran Yahav, ICCAV 2018.
- <img src="badges/13-pages-gray.svg" alt="13-pages" align="top"> [Neural Code Comprehension: A Learnable Representation of Code Semantics](https://arxiv.org/abs/1806.07336) - Tal Ben-Nun, Alice Shoshana Jakobovits, Torsten Hoefler, NIPS 2018.
- <img src="badges/16-pages-gray.svg" alt="16-pages" align="top"> [A General Path-Based Representation for Predicting Program Properties](https://arxiv.org/abs/1803.09544) - Uri Alon, Meital Zilberstein, Omer Levy, Eran Yahav, PLDI 2018.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [Cross-Language Learning for Program Classification using Bilateral Tree-Based Convolutional Neural Networks](https://arxiv.org/abs/1710.06159v2) - Nghi D. Q. Bui, Lingxiao Jiang, Yijun Yu, AAAI 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Bilateral Dependency Neural Networks for Cross-Language Algorithm Classification](https://bdqnghi.github.io/files/SANER_2019_bilateral_dependency.pdf) - Nghi D. Q. Bui, Yijun Yu, Lingxiao Jiang, SANER 2018.
- <img src="badges/17-pages-gray.svg" alt="17-pages" align="top"> [Syntax-Directed Variational Autoencoder for Structured Data](https://openreview.net/pdf?id=SyqShMZRb) - Hanjun Dai, Yingtao Tian, Bo Dai, Steven Skiena, Le Song, ICLR 2018.
- <img src="badges/19-pages-gray.svg" alt="19-pages" align="top"> [Divide and Conquer with Neural Networks](https://arxiv.org/abs/1611.02401) - Nowak, Alex, and Joan Bruna, ICLR 2018.
- <img src="badges/13-pages-gray.svg" alt="13-pages" align="top"> [Hierarchical multiscale recurrent neural networks](https://arxiv.org/abs/1609.01704) - Chung Junyoung, Sungjin Ahn, and Yoshua Bengio, ICLR 2017.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Learning Efficient Algorithms with Hierarchical Attentive Memory](https://arxiv.org/abs/1602.03218) - Andrychowicz, Marcin, and Karol Kurach, 2016.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [Learning Operations on a Stack with Neural Turing Machines](https://arxiv.org/abs/1612.00827) - Deleu, Tristan, and Joseph Dureau, NIPS 2016.
- <img src="badges/5-pages-gray.svg" alt="5-pages" align="top"> [Probabilistic Neural Programs](https://arxiv.org/abs/1612.00712) - Murray, Kenton W., and Jayant Krishnamurthy, NIPS 2016.
- <img src="badges/13-pages-gray.svg" alt="13-pages" align="top"> [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279) - Reed, Scott, and Nando de Freitas, ICLR 2016.
- <img src="badges/9-pages-gray.svg" alt="9-pages" align="top"> [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) - Kaiser, Łukasz, and Ilya Sutskever, ICLR 2016.
- <img src="badges/17-pages-gray.svg" alt="17-pages" align="top"> [Neural Random-Access Machines](https://arxiv.org/abs/1511.06392v3) - Karol Kurach, Marcin Andrychowicz, Ilya Sutskever, ERCIM News 2016.
- <img src="badges/18-pages-gray.svg" alt="18-pages" align="top"> [Neural Programmer: Inducing Latent Programs with Gradient Descent](https://arxiv.org/abs/1511.04834) - Neelakantan, Arvind, Quoc V. Le, and Ilya Sutskever, ICLR 2015.
- <img src="badges/25-pages-gray.svg" alt="25-pages" align="top"> [Learning to Execute](https://arxiv.org/abs/1410.4615v3) - Wojciech Zaremba, Ilya Sutskever, 2015.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](https://arxiv.org/abs/1503.01007) - Joulin, Armand, and Tomas Mikolov, NIPS 2015.
- <img src="badges/26-pages-gray.svg" alt="26-pages" align="top"> [Neural Turing Machines](https://arxiv.org/abs/1410.5401) - Graves, Alex, Greg Wayne, and Ivo Danihelka, 2014.
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [From Machine Learning to Machine Reasoning](https://arxiv.org/abs/1102.1808) - Bottou Leon, Journal of Machine Learning 2011.

#### Embeddings in Software Engineering

- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [A Literature Study of Embeddings on Source Code](https://arxiv.org/abs/1904.03061) - Zimin Chen and Martin Monperrus, 2019.
- <img src="badges/3-pages-gray.svg" alt="3-pages" align="top"> [AST-Based Deep Learning for Detecting Malicious PowerShell](https://arxiv.org/pdf/1810.09230.pdf) - Gili Rusak, Abdullah Al-Dujaili, Una-May O'Reilly, 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Deep Code Search](https://dl.acm.org/citation.cfm?id=3180167) - Xiaodong Gu, Hongyu Zhang, Sunghun Kim, ICSE 2018.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [Word Embeddings for the Software Engineering Domain](https://github.com/vefstathiou/SO_word2vec/blob/master/MSR18-w2v.pdf) - Vasiliki Efstathiou, Christos Chatzilenas, Diomidis Spinellis, MSR 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align=top> [
  Code Vectors: Understanding Programs Through Embedded Abstracted Symbolic Traces](https://arxiv.org/abs/1803.06686) - Jordan Henkel, Shuvendu K. Lahiri, Ben Liblit, Thomas Reps, FSE 2018.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Document Distance Estimation via Code Graph Embedding](https://www.researchgate.net/publication/320074701_Document_Distance_Estimation_via_Code_Graph_Embedding) - Zeqi Lin, Junfeng Zhao, Yanzhen Zou, Bing Xie, Internetware 2017.
- <img src="badges/3-pages-gray.svg" alt="3-pages" align="top"> [Combining Word2Vec with revised vector space model for better code retrieval](https://www.researchgate.net/publication/318123700_Combining_Word2Vec_with_Revised_Vector_Space_Model_for_Better_Code_Retrieval) - Thanh Van Nguyen, Anh Tuan Nguyen, Hung Dang Phan, Trong Duc Nguyen, Tien N. Nguyen, ICSE 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [From word embeddings to document similarities for improved information retrieval in software engineering](https://www.researchgate.net/publication/296526040_From_Word_Embeddings_To_Document_Similarities_for_Improved_Information_Retrieval_in_Software_Engineering) - Xin Ye, Hui Shen, Xiao Ma, Razvan Bunescu, Chang Liu, ICSE 2016.
- <img src="badges/3-pages-gray.svg" alt="3-pages" align="top"> [Mapping API Elements for Code Migration with Vector Representation](https://dl.acm.org/citation.cfm?id=2892661) - Trong Duc Nguyen, Anh Tuan Nguyen, Tien N. Nguyen, ICSE 2016.

#### Program Translation

- <img src="badges/18-pages-gray.svg" alt="18-pages" align="top"> [Towards Neural Decompilation](https://arxiv.org/abs/1905.08325v1) - Omer Katz, Yuval Olshaker, Yoav Goldberg, Eran Yahav, 2019.
- <img src="badges/14-pages-gray.svg" alt="14-pages" align="top"> [Tree-to-tree Neural Networks for Program Translation](https://arxiv.org/abs/1802.03691v1) - Xinyun Chen, Chang Liu, Dawn Song, ICLR 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Code Attention: Translating Code to Comments by Exploiting Domain Features](https://arxiv.org/abs/1709.07642v2) - Wenhao Zheng, Hong-Yu Zhou, Ming Li, Jianxin Wu, 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Automatically Generating Commit Messages from Diffs using Neural Machine Translation](https://arxiv.org/abs/1708.09492v1) - Siyuan Jiang, Ameer Armaly, Collin McMillan, ASE 2017.
- <img src="badges/5-pages-gray.svg" alt="5-pages" align="top"> [A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation](https://arxiv.org/abs/1707.02275v1) - Antonio Valerio Miceli Barone, Rico Sennrich, ICNLP 2017.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [A Neural Architecture for Generating Natural Language Descriptions from Source Code Changes](https://arxiv.org/abs/1704.04856v1) - Pablo Loyola, Edison Marrese-Taylor, Yutaka Matsuo, ACL 2017.

#### Code Suggestion and Completion

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Aroma: Code Recommendation via Structural Code Search](https://arxiv.org/abs/1812.01158) - Sifei Luan, Di Yang, Koushik Sen and Satish Chandra, 2019.
- <img src="badges/9-pages-gray.svg" alt="9-pages" align="top"> [Intelligent Code Reviews Using Deep Learning](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_40.pdf) - Anshul Gupta, Neel Sundaresan, KDD DL Day 2018.
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [Code Completion with Neural Attention and Pointer Networks](https://arxiv.org/abs/1711.09573v1) - Jian Li, Yue Wang, Irwin King, Michael R. Lyu, 2017.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Learning Python Code Suggestion with a Sparse Pointer Network](https://arxiv.org/abs/1611.08307) - Avishkar Bhoopchand, Tim Rocktäschel, Earl Barr, Sebastian Riedel, 2016.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Code Completion with Statistical Language Models](http://www.cs.technion.ac.il/~yahave/papers/pldi14-statistical.pdf) - Veselin Raychev, Martin Vechev, Eran Yahav, PLDI 2014.

#### Program Repair and Bug Detection

- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [SampleFix: Learning to Correct Programs by Sampling Diverse Fixes](https://arxiv.org/abs/1906.10502) - Hossein Hajipour, Apratim Bhattacharya, Mario Fritz, 2019.
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [Maximal Divergence Sequential Autoencoder for Binary Software Vulnerability Detection](https://openreview.net/forum?id=ByloIiCqYQ) - Tue Le, Tuan Nguyen, Trung Le, Dinh Phung, Paul Montague, Olivier De Vel, Lizhen Qu, ICLR 2019.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Neural Program Repair by Jointly Learning to Localize and Repair](https://openreview.net/forum?id=ByloJ20qtm) - Marko Vasic, Aditya Kanade, Petros Maniatis, David Bieber, Rishabh Singh, ICLR 2019.
- <img src="badges/11-pages-beginner-brightgreen.svg" alt="11-pages" align="top"> [Compiler Fuzzing through Deep Learning](https://chriscummins.cc/pub/2018-issta.pdf) - Chris Cummins, Pavlos Petoumenos, Alastair Murray, Hugh Leather, ISSTA 2018
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Automatically assessing vulnerabilities discovered by compositional analysis](https://dl.acm.org/citation.cfm?id=3243130) - Saahil Ognawala, Ricardo Nales Amato, Alexander Pretschner and Pooja Kulkarni, MASES 2018.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [An Empirical Investigation into Learning Bug-Fixing Patches in the Wild via Neural Machine Translation](http://www.cs.wm.edu/~denys/pubs/ASE%2718-Learning-Bug-Fixes-NMT.pdf) - Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin White, Denys Poshyvanyk, ASE 2018.
- <img src="badges/23-pages-gray.svg" alt="23-pages" align="top"> [DeepBugs: A Learning Approach to Name-based Bug Detection](https://arxiv.org/pdf/1805.11683.pdf) - Michael Pradel, Koushik Sen, 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Learning How to Mutate Source Code from Bug-Fixes](https://arxiv.org/abs/1812.10772) - Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin White, Denys Poshyvanyk, 2018.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [A deep tree-based model for software defect prediction](https://arxiv.org/abs/1802.00921) - HK Dam, T Pham, SW Ng, [T Tran](https://truyentran.github.io), J Grundy, A Ghose, T Kim, CJ Kim, 2018.
- <img src="badges/7-pages-gray.svg" alt="7-pages" align="top"> [Automated Vulnerability Detection in Source Code Using Deep Representation Learning](https://arxiv.org/abs/1807.04320) - Rebecca L. Russell, Louis Kim, Lei H. Hamilton, Tomo Lazovich, Jacob A. Harer, Onur Ozdemir, Paul M. Ellingwood, Marc W. McConley, 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Shaping Program Repair Space with Existing Patches and Similar Code](https://xiongyingfei.github.io/papers/ISSTA18a.pdf) - Jiajun Jiang, Yingfei Xiong, Hongyu Zhang, Qing Gao, Xiangqun Chen, 2018. ([code](https://github.com/xgdsmileboy/SimFix)).
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [Learning to Repair Software Vulnerabilities with Generative Adversarial Networks](https://arxiv.org/abs/1805.07475) - Jacob A. Harer, Onur Ozdemir, Tomo Lazovich, Christopher P. Reale, Rebecca L. Russell, Louis Y. Kim, Peter Chin, 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Dynamic Neural Program Embedding for Program Repair](https://arxiv.org/abs/1711.07163v2) - Ke Wang, Rishabh Singh, Zhendong Su, ICLR 2018.
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [Estimating defectiveness of source code: A predictive model using GitHub content](https://arxiv.org/abs/1803.07764) - Ritu Kapur, Balwinder Sodhi, 2018
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [Automated software vulnerability detection with machine learning](https://arxiv.org/abs/1803.04497) - Jacob A. Harer, Louis Y. Kim, Rebecca L. Russell, Onur Ozdemir, Leonard R. Kosta, Akshay Rangamani, Lei H. Hamilton, Gabriel I. Centeno, Jonathan R. Key, Paul M. Ellingwood, Marc W. McConley, Jeffrey M. Opper, Peter Chin, Tomo Lazovich, IWSPA 2018.
- <img src="badges/34-pages-gray.svg" alt="34-pages" align="top"> [Learning a Static Analyzer from Data](https://arxiv.org/abs/1611.01752) - Pavol Bielik, Veselin Raychev, Martin Vechev, CAV 2017. [video](https://www.youtube.com/watch?v=bkieI3jLxVY).
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [To Type or Not to Type: Quantifying Detectable Bugs in JavaScript](http://earlbarr.com/publications/typestudy.pdf) - Zheng Gao, Christian Bird, Earl Barr, ICSE 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Sorting and Transforming Program Repair Ingredients via Deep Learning Code Similarities](https://arxiv.org/abs/1707.04742) - Martin White, Michele Tufano, Matías Martínez, Martin Monperrus, Denys Poshyvanyk, 2017.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Semantic Code Repair using Neuro-Symbolic Transformation Networks](https://arxiv.org/abs/1710.11054v1) - Jacob Devlin, Jonathan Uesato, Rishabh Singh, Pushmeet Kohli, 2017.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [Automated Identification of Security Issues from Commit Messages and Bug Reports](http://asankhaya.github.io/pdf/automated-identification-of-security-issues-from-commit-messages-and-bug-reports.pdf) - Yaqin Zhou and Asankhaya Sharma, FSE 2017.
- <img src="badges/31-pages-gray.svg" alt="31-pages" align="top"> [SmartPaste: Learning to Adapt Source Code](https://arxiv.org/abs/1705.07867) - Miltiadis Allamanis, Marc Brockschmidt, 2017.
- <img src="badges/7-pages-gray.svg" alt="7-pages" align="top"> [End-to-End Prediction of Buffer Overruns from Raw Source Code via Neural Memory Networks](https://arxiv.org/abs/1703.02458v1) - Min-je Choi, Sehun Jeong, Hakjoo Oh, Jaegul Choo, IJCAI 2017.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Tailored Mutants Fit Bugs Better](https://arxiv.org/abs/1611.02516) - Miltiadis Allamanis, Earl T. Barr, René Just, Charles Sutton, 2016.

#### APIs and Code Mining

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [SAR: Learning Cross-Language API Mappings with Little Knowledge](https://bdqnghi.github.io/files/FSE_2019.pdf) - Nghi D. Q. Bui, Yijun Yu, Lingxiao Jiang, FSE 2019.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [Hierarchical Learning of Cross-Language Mappings through Distributed Vector Representations for Code](https://arxiv.org/abs/1803.04715) - Nghi D. Q. Bui, Lingxiao Jiang, ICSE 2018.
- <img src="badges/7-pages-gray.svg" alt="7-pages" align="top"> [DeepAM: Migrate APIs with Multi-modal Sequence to Sequence Learning](https://arxiv.org/abs/1704.07734v1) - Xiaodong Gu, Hongyu Zhang, Dongmei Zhang, Sunghun Kim, IJCAI 2017.
- <img src="badges/9-pages-gray.svg" alt="9-pages" align="top"> [Mining Change Histories for Unknown Systematic Edits](http://soft.vub.ac.be/Publications/2017/vub-soft-tr-17-04.pdf) - Tim Molderez, Reinout Stevens, Coen De Roover, MSR 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Deep API Learning](https://arxiv.org/abs/1605.08535v3) - Xiaodong Gu, Hongyu Zhang, Dongmei Zhang, Sunghun Kim, FSE 2016.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Exploring API Embedding for API Usages and Applications](http://home.eng.iastate.edu/~trong/projects/jv2cs/) - Nguyen, Nguyen, Phan and Nguyen, Journal of Systems and Software 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [API usage pattern recommendation for software development](http://www.sciencedirect.com/science/article/pii/S0164121216301200) - Haoran Niu, Iman Keivanloo, Ying Zou, 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Parameter-Free Probabilistic API Mining across GitHub](http://homepages.inf.ed.ac.uk/csutton/publications/fse2016.pdf) - Jaroslav Fowkes, Charles Sutton, FSE 2016.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [A Subsequence Interleaving Model for Sequential Pattern Mining](http://homepages.inf.ed.ac.uk/csutton/publications/kdd2016-subsequence-interleaving.pdf) - Jaroslav Fowkes, Charles Sutton, KDD 2016.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [Lean GHTorrent: GitHub data on demand](https://bvasiles.github.io/papers/lean-ghtorrent.pdf) - Georgios Gousios, Bogdan Vasilescu, Alexander Serebrenik, Andy Zaidman, MSR 2014.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Mining idioms from source code](http://homepages.inf.ed.ac.uk/csutton/publications/idioms.pdf) - Miltiadis Allamanis, Charles Sutton, FSE 2014.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [The GHTorent Dataset and Tool Suite](http://www.gousios.gr/pub/ghtorrent-dataset-toolsuite.pdf) - Georgios Gousios, MSR 2013.

#### Code Optimization

- <img src="badges/27-pages-gray.svg" alt="27-pages" align="top"> [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208v2) - Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, Neoklis Polyzotis, SIGMOD 2018.
- <img src="badges/14-pages-gray.svg" alt="14-pages" align="top"> [End-to-end Deep Learning of Optimization Heuristics](https://chriscummins.cc/pub/2017-pact.pdf) - Chris Cummins, Pavlos Petoumenos, Zheng Wang, Hugh Leather, PACT 2017
- <img src="badges/14-pages-gray.svg" alt="14-pages" align="top"> [Learning to superoptimize programs](https://arxiv.org/abs/1611.01787v3) - Rudy Bunel, Alban Desmaison, M. Pawan Kumar, Philip H.S. Torr, Pushmeet Kohlim ICLR 2017.
- <img src="badges/18-pages-gray.svg" alt="18-pages" align="top"> [Neural Nets Can Learn Function Type Signatures From Binaries](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-chua.pdf) - Zheng Leong Chua, Shiqi Shen, Prateek Saxena, and Zhenkai Liang, USENIX Security Symposium 2017.
- <img src="badges/25-pages-gray.svg" alt="25-pages" align="top"> [Adaptive Neural Compilation](https://arxiv.org/abs/1605.07969v2) - Rudy Bunel, Alban Desmaison, Pushmeet Kohli, Philip H.S. Torr, M. Pawan Kumar, NIPS 2016.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Learning to Superoptimize Programs - Workshop Version](https://arxiv.org/abs/1612.01094) - Bunel, Rudy, Alban Desmaison, M. Pawan Kumar, Philip H. S. Torr, and Pushmeet Kohli, NIPS 2016.

#### Topic Modeling

- <img src="badges/9-pages-gray.svg" alt="9-pages" align="top"> [A Language-Agnostic Model for Semantic Source Code Labeling](https://dl.acm.org/citation.cfm?id=3243132) - Ben Gelman, Bryan Hoyle, Jessica Moore, Joshua Saxe and David Slater, MASES 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Topic modeling of public repositories at scale using names in source code](https://arxiv.org/abs/1704.00135) - Vadim Markovtsev, Eiso Kant, 2017.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [Why, When, and What: Analyzing Stack Overflow Questions by Topic, Type, and Code](http://homepages.inf.ed.ac.uk/csutton/publications/msrCh2013.pdf) - Miltiadis Allamanis, Charles Sutton, MSR 2013.
- <img src="badges/30-pages-gray.svg" alt="30-pages" align="top"> [Semantic clustering: Identifying topics in source code](http://scg.unibe.ch/archive/drafts/Kuhn06bSemanticClustering.pdf) - Adrian Kuhn, Stéphane Ducasse, Tudor Girba, Information & Software Technology 2007.

#### Sentiment Analysis

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [A Benchmark Study on Sentiment Analysis for Software Engineering Research](https://arxiv.org/abs/1803.06525) - Nicole Novielli, Daniela Girardi, Filippo Lanubile, MSR 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Sentiment Analysis for Software Engineering: How Far Can We Go?](http://www.inf.usi.ch/phd/lin/downloads/Lin2018a.pdf) - Bin Lin, Fiorella Zampetti, Gabriele Bavota, Massimiliano Di Penta, Michele Lanza, Rocco Oliveto, ICSE 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Leveraging Automated Sentiment Analysis in Software Engineering](http://cs.uno.edu/~zibran/resources/MyPapers/SentiStrengthSE_2017.pdf) - Md Rakibul Islam, Minhaz F. Zibran, MSR 2017.
- <img src="badges/27-pages-gray.svg" alt="27-pages" align="top"> [Sentiment Polarity Detection for Software Development](https://arxiv.org/pdf/1709.02984.pdf) - Fabio Calefato, Filippo Lanubile, Federico Maiorano, Nicole Novielli, Empirical Software Engineering 2017.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [SentiCR: A Customized Sentiment Analysis Tool for Code Review Interactions](https://drive.google.com/file/d/0Byog0ILN8S1haGxpT3hvSzZxdms/view) - Toufique Ahmed, Amiangshu Bosu, Anindya Iqbal, Shahram Rahimi, ASE 2017.

#### Code Summarization

- <img src="badges/7-pages-gray.svg" alt="7-pages" align="top"> [Summarizing Source Code with Transferred API Knowledge](https://xin-xia.github.io/publication/ijcai18.pdf) - Xing Hu, Ge Li, Xin Xia, David Lo, Shuai Lu, Zhi Jin, IJCAI 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Deep Code Comment Generation](https://xin-xia.github.io/publication/icpc182.pdf) - Xing Hu, Ge Li, Xin Xia, David Lo, Zhi Jin, ICPC 2018.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [A Neural Framework for Retrieval and Summarization of Source Code](https://dl.acm.org/citation.cfm?id=3240471) - Qingying Chen, Minghui Zhou, ASE 2018.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Improving Automatic Source Code Summarization via Deep Reinforcement Learning](https://arxiv.org/abs/1811.07234) - Yao Wan, Zhou Zhao, Min Yang, Guandong Xu, Haochao Ying, Jian Wu and Philip S. Yu, ASE 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [A Convolutional Attention Network for Extreme Summarization of Source Code](https://arxiv.org/abs/1602.03001) - Miltiadis Allamanis, Hao Peng, Charles Sutton, ICML 2016.
- <img src="badges/4-pages-gray.svg" alt="4-pages" align="top"> [TASSAL: Autofolding for Source Code Summarization](http://homepages.inf.ed.ac.uk/csutton/publications/icse2016-demo.pdf) - Jaroslav Fowkes, Pankajan Chanthirasegaran, Razvan Ranca, Miltiadis Allamanis, Mirella Lapata, Charles Sutton, ICSE 2016.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Summarizing Source Code using a Neural Attention Model](https://github.com/sriniiyer/codenn/blob/master/summarizing_source_code.pdf) - Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, Luke Zettlemoyer, ACL 2016.
- <img src="badges/13-pages-gray.svg" alt="13-pages" align="top"> [Automatic Generation of Pull Request Descriptions](https://arxiv.org/abs/1909.06987) - Zhongxin Liu, Xin Xia, Christoph Treude, David Lo, Shanping Li, ASE 2019.

#### Clone Detection

- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Learning-Based Recursive Aggregation of Abstract Syntax Trees for Code Clone Detection](https://pvs.ifi.uni-heidelberg.de/fileadmin/papers/2019/Buech-Andrzejak-SANER2019.pdf) - Lutz Büch and Artur Andrzejak, SANER 2019.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Oreo: detection of clones in the twilight zone](https://dl.acm.org/citation.cfm?id=3236026) - Vaibhav Saini, Farima Farmahinifarahani, Yadong Lu, Pierre Baldi, and Cristina V. Lopes, FSE 2018.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [A Deep Learning Approach to Program Similarity](https://dl.acm.org/citation.cfm?id=3243131) - Niccolò Marastoni, Roberto Giacobazzi and Mila Dalla Preda, MASES 2018.
- <img src="badges/6-pages-gray.svg" alt="6-pages" align="top"> [Recurrent Neural Network for Code Clone Detection](https://seim-conf.org/media/materials/2018/proceedings/SEIM-2018_Short_Papers.pdf#page=48) - Arseny Zorin and Vladimir Itsykson, SEIM 2018.
- <img src="badges/8-pages-gray.svg" alt="8-pages" align="top"> [The Adverse Effects of Code Duplication in Machine Learning Models of Code](https://arxiv.org/abs/1812.06469) - Miltiadis Allamanis, 2018.
- <img src="badges/28-pages-gray.svg" alt="28-pages" align="top"> [DéjàVu: a map of code duplicates on GitHub](http://janvitek.org/pubs/oopsla17b.pdf) - Cristina V. Lopes, Petr Maj, Pedro Martins, Vaibhav Saini, Di Yang, Jakub Zitny, Hitesh Sajnani, Jan Vitek, Programming Languages OOPSLA 2017.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Some from Here, Some from There: Cross-project Code Reuse in GitHub](http://web.cs.ucdavis.edu/~filkov/papers/clones.pdf) - Mohammad Gharehyazie, Baishakhi Ray, Vladimir Filkov, MSR 2017.
- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [Deep Learning Code Fragments for Code Clone Detection](http://www.cs.wm.edu/~denys/pubs/ASE%2716-DeepLearningClones.pdf) - Martin White, Michele Tufano, Christopher Vendome, and Denys Poshyvanyk, ASE 2016.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [A study of repetitiveness of code changes in software evolution](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?referer=https://scholar.google.com/&httpsredir=1&article=1016&context=cs_conf) - HA Nguyen, AT Nguyen, TT Nguyen, TN Nguyen, H Rajan, ASE 2013.

#### Differentiable Interpreters

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [DDRprog: A CLEVR Differentiable Dynamic Reasoning Programmer](https://arxiv.org/abs/1803.11361v1) - Joseph Suarez, Justin Johnson, Fei-Fei Li, 2018.
- <img src="badges/16-pages-gray.svg" alt="16-pages" align="top"> [Improving the Universality and Learnability of Neural Programmer-Interpreters with Combinator Abstraction](https://arxiv.org/abs/1802.02696v1) - Da Xiao, Jo-Yu Liao, Xingyuan Yuan, ICLR 2018.
- <img src="badges/10-pages-gray.svg" alt="10-pages" align="top"> [Differentiable Programs with Neural Libraries](https://arxiv.org/abs/1611.02109v2) - Alexander L. Gaunt, Marc Brockschmidt, Nate Kushman, Daniel Tarlow, ICML 2017.
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [Differentiable Functional Program Interpreters](https://arxiv.org/abs/1611.01988v2) - John K. Feser, Marc Brockschmidt, Alexander L. Gaunt, Daniel Tarlow, 2017.
- <img src="badges/18-pages-gray.svg" alt="18-pages" align="top"> [Programming with a Differentiable Forth Interpreter](https://arxiv.org/abs/1605.06640) - Bošnjak, Matko, Tim Rocktäschel, Jason Naradowsky, and Sebastian Riedel, ICML 2017.
- <img src="badges/15-pages-gray.svg" alt="15-pages" align="top"> [Neural Functional Programming](https://arxiv.org/abs/1611.01988v1) - Feser John K., Marc Brockschmidt, Alexander L. Gaunt, and Daniel Tarlow, ICLR 2017.
- <img src="badges/7-pages-gray.svg" alt="7-pages" align="top"> [TerpreT: A Probabilistic Programming Language for Program Induction](https://arxiv.org/abs/1612.00817) - Gaunt, Alexander L., Marc Brockschmidt, Rishabh Singh, Nate Kushman, Pushmeet Kohli, Jonathan Taylor, and Daniel Tarlow, NIPS 2016.

<a name="related-research"></a>

<details>
<summary>Related research</summary>

#### AST Differencing

- <img src="badges/12-pages-gray.svg" alt="12-pages" align="top"> [ClDiff: Generating Concise Linked Code Differences](https://chenbihuan.github.io/paper/ase18-huang-cldiff.pdf) - Kaifeng Huang, Bihuan Chen, Xin Peng, Daihong Zhou, Ying Wang, Yang Liu, Wenyun Zhao, ASE 2018. [Code](https://github.com/FudanSELab/CLDIFF).
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Generating Accurate and Compact Edit Scripts Using Tree Differencing](http://www.xifiggam.eu/wp-content/uploads/2018/08/GeneratingAccurateandCompactEditScriptsusingTreeDifferencing.pdf) - Veit Frick, Thomas Grassauer, Fabian Beck, Martin Pinzger, ICSME 2018.
- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [Fine-grained and Accurate Source Code Differencing](https://hal.archives-ouvertes.fr/hal-01054552/document) - Jean-Rémy Falleri, Floréal Morandat, Xavier Blanc, Matias Martinez, Martin Monperrus, ASE 2014.

#### Binary Data Modeling

- [Clustering Binary Data with Bernoulli Mixture Models](https://nsgrantham.com/documents/clustering-binary-data.pdf) - Neal S. Grantham.
- [A Family of Blockwise One-Factor Distributions for Modelling High-Dimensional Binary Data](https://arxiv.org/pdf/1511.01343.pdf) - Matthieu Marbac and Mohammed Sedki, Computational Statistics & Data Analysis 2017.
- [BayesBinMix: an R Package for Model Based Clustering of Multivariate Binary Data](https://arxiv.org/pdf/1609.06960.pdf) - Panagiotis Papastamoulis and Magnus Rattray, R Journal 2016.

#### Soft Clustering Using T-mixture Models

- [Robust mixture modelling using the t distribution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.7334&rep=rep1&type=pdf) - D. Peel and G. J. McLachlan, Statistics and Computing 2000.
- [Robust mixture modeling using the skew t distribution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1030.9865&rep=rep1&type=pdf) - Tsung I. Lin, Jack C. Lee and Wan J. Hsieh, Statistics and Computing 2010.

#### Natural Language Parsing and Comprehension

- <img src="badges/11-pages-gray.svg" alt="11-pages" align="top"> [A Fast Unified Model for Parsing and Sentence Understanding](https://arxiv.org/abs/1603.06021) - Samuel R. Bowman, Jon Gauthier, Abhinav Rastogi, Raghav Gupta, Christopher D. Manning, Christopher Potts, ACL 2016.

</details>

## Posts

- [Semantic Code Search](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)
- [Learning from Source Code](https://www.microsoft.com/en-us/research/blog/learning-source-code/)
- [Training a Model to Summarize Github Issues](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)
- [Sequence Intent Classification Using Hierarchical Attention Networks](https://www.microsoft.com/developerblog/2018/03/06/sequence-intent-classification/)
- [Syntax-Directed Variational Autoencoder for Structured Data](https://mlatgt.blog/2018/02/08/syntax-directed-variational-autoencoder-for-structured-data/)
- [Weighted MinHash on GPU helps to find duplicate GitHub repositories.](https://blog.sourced.tech//post/minhashcuda/)
- [Source Code Identifier Embeddings](https://blog.sourced.tech/post/id2vec/)
- [Using recurrent neural networks to predict next tokens in the java solutions](https://codeforces.com/blog/entry/52327)
- [The half-life of code & the ship of Theseus](https://erikbern.com/2016/12/05/the-half-life-of-code.html)
- [The eigenvector of "Why we moved from language X to language Y"](https://erikbern.com/2017/03/15/the-eigenvector-of-why-we-moved-from-language-x-to-language-y.html)
- [Analyzing Github, How Developers Change Programming Languages Over Time](https://blog.sourced.tech/post/language_migrations/)
- [Topic Modeling of GitHub Repositories](https://blog.sourced.tech//post/github_topic_modeling/)
- [Aroma: Using machine learning for code recommendation](https://ai.facebook.com/blog/aroma-ml-for-code-recommendation/)

## Talks

- [Machine Learning on Source Code](http://vmarkovtsev.github.io/pydays-2018-vienna/)
- [Similarity of GitHub Repositories by Source Code Identifiers](http://vmarkovtsev.github.io/techtalks-2017-moscow/)
- [Using deep RNN to model source code](http://vmarkovtsev.github.io/re-work-2016-london/)
- [Source code abstracts classification using CNN (1)](http://vmarkovtsev.github.io/re-work-2016-berlin/)
- [Source code abstracts classification using CNN (2)](http://vmarkovtsev.github.io/data-natives-2016/)
- [Source code abstracts classification using CNN (3)](http://vmarkovtsev.github.io/slush-2016/)
- [Embedding the GitHub contribution graph](https://egorbu.github.io/techtalks-2017-moscow)
- [Measuring code sentiment in a Git repository](http://vmarkovtsev.github.io/gophercon-2018-moscow/)

## Software

#### Machine Learning

- [Differentiable Neural Computer (DNC)](https://github.com/deepmind/dnc) - TensorFlow implementation of the Differentiable Neural Computer.
- [sourced.ml](https://github.com/src-d/ml) - Abstracts feature extraction from source code syntax trees and working with ML models.
- [vecino](https://github.com/src-d/vecino) - Finds similar Git repositories.
- [apollo](https://github.com/src-d/apollo) - Source code deduplication as scale, research.
- [gemini](https://github.com/src-d/gemini) - Source code deduplication as scale, production.
- [enry](https://github.com/src-d/enry) - Insanely fast file based programming language detector.
- [hercules](https://github.com/src-d/hercules) - Git repository mining framework with batteries on top of go-git.
- [DeepCS](https://github.com/guxd/deep-code-search) - Keras and Pytorch implementations of DeepCS (Deep Code Search).
- [Code Neuron](https://github.com/vmarkovtsev/codeneuron) - Recurrent neural network to detect code blocks in natural language text.
- [Naturalize](https://github.com/mast-group/naturalize) - Language agnostic framework for learning coding conventions from a codebase and then expoiting this information for suggesting better identifier names and formatting changes in the code.
- [Extreme Source Code Summarization](https://github.com/mast-group/convolutional-attention) - Convolutional attention neural network that learns to summarize source code into a short method name-like summary by just looking at the source code tokens.
- [Summarizing Source Code using a Neural Attention Model](https://github.com/sriniiyer/codenn) - CODE-NN, uses LSTM networks with attention to produce sentences that describe C# code snippets and SQL queries from StackOverflow. Torch over C#/SQL
- [Probabilistic API Miner](https://github.com/mast-group/api-mining) - Near parameter-free probabilistic algorithm for mining the most interesting API patterns from a list of API call sequences.
- [Interesting Sequence Miner](https://github.com/mast-group/sequence-mining) - Novel algorithm that mines the most interesting sequences under a probabilistic model. It is able to efficiently infer interesting sequences directly from the database.
- [TASSAL](https://github.com/mast-group/tassal) - Tool for the automatic summarization of source code using autofolding. Autofolding automatically creates a summary of a source code file by folding non-essential code and comment blocks.
- [JNice2Predict](http://www.nice2predict.org/) - Efficient and scalable open-source framework for structured prediction, enabling one to build new statistical engines more quickly.
- [Clone Digger](http://clonedigger.sourceforge.net/download.html) - clone detection for Python and Java.
- [Sensibility](https://github.com/naturalness/sensibility) - Uses LSTMs to detect and correct syntax errors in Java source code.
- [DeepBugs](https://github.com/michaelpradel/DeepBugs) - Framework for learning bug detectors from an existing code corpus.
- [DeepSim](https://github.com/parasol-aser/deepsim) - a deep learning-based approach to measure code functional similarity.
- [rnn-autocomplete](https://github.com/ZeRoGerc/rnn-autocomplete) - Neural code autocompletion with RNN (bachelor's thesis).
- [MindsDB](https://github.com/mindsdb/mindsdb) - MindsDB is an Explainable AutoML framework for developers. With MindsDB you can build, train and use state of the art ML models in as simple as one line of code.

#### Utilities

- [go-git](https://github.com/src-d/go-git) - Highly extensible Git implementation in pure Go which is friendly to data mining.
- [bblfsh](https://github.com/bblfsh) - Self-hosted server for source code parsing.
- [engine](https://github.com/src-d/engine) - Scalable and distributed data retrieval pipeline for source code.
- [minhashcuda](https://github.com/src-d/minhashcuda) - Weighted MinHash implementation on CUDA to efficiently find duplicates.
- [kmcuda](https://github.com/src-d/kmcuda) - k-means on CUDA to cluster and to search for nearest neighbors in dense space.
- [wmd-relax](https://github.com/src-d/wmd-relax) - Python package which finds nearest neighbors at Word Mover's Distance.
- [Tregex, Tsurgeon and Semgrex](https://nlp.stanford.edu/software/tregex.shtml) - Tregex is a utility for matching patterns in trees, based on tree relationships and regular expression matches on nodes (the name is short for "tree regular expressions").
- [source{d} models](https://github.com/src-d/models) - Machine Learning models for MLonCode trained using the source{d} stack.

#### Datasets

- [Neural-Code-Search-Evaluation-Dataset](https://github.com/facebookresearch/Neural-Code-Search-Evaluation-Dataset) - dataset contains links to 4.7M methods from 24k+ repositories with 287 StackOverflow questions and code snippet answers.
- [CodeSearchNet](https://github.com/github/CodeSearchNet) -  collection of datasets and benchmarks for code retrieval using natural language. Contains 2M pairs of (`comment`, `code`).
- [Public Git Archive](https://github.com/src-d/datasets/tree/master/PublicGitArchive) - 6 TB of Git repositories from GitHub.
- [StackOverflow Question-Code Dataset](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) - ~148K Python and ~120K SQL question-code pairs mined from StackOverflow.
- [GitHub Issue Titles and Descriptions for NLP Analysis](https://www.kaggle.com/davidshinn/github-issues/) - ~8 million GitHub issue titles and descriptions from 2017.
- [GitHub repositories - languages distribution](https://data.world/source-d/github-repositories-languages-distribution) - Programming languages distribution in 14,000,000 repositories on GitHub (October 2016).
- [452M commits on GitHub](https://data.world/vmarkovtsev/452-m-commits-on-github) - ≈ 452M commits' metadata from 16M repositories on GitHub (October 2016).
- [GitHub readme files](https://data.world/vmarkovtsev/github-readme-files) - Readme files of all GitHub repositories (16M) (October 2016).
- [from language X to Y](https://data.world/vmarkovtsev/from-language-x-to-y) - Cache file Erik Bernhardsson collected for his awesome blog post.
- [GitHub word2vec 120k](https://data.world/vmarkovtsev/github-word-2-vec-120-k) - Sequences of identifiers extracted from top starred 120,000 GitHub repositories.
- [GitHub Source Code Names](https://data.world/vmarkovtsev/github-source-code-names) - Names in source code extracted from 13M GitHub repositories, not people.
- [GitHub duplicate repositories](https://data.world/vmarkovtsev/github-duplicate-repositories) - GitHub repositories not marked as forks but very similar to each other.
- [GitHub lng keyword frequencies](https://data.world/vmarkovtsev/github-lng-keyword-frequencies) - Programming language keyword frequency extracted from 16M GitHub repositories.
- [GitHub Java Corpus](http://groups.inf.ed.ac.uk/cup/javaGithub/) - GitHub Java corpus is a set of Java projects collected from GitHub that we have used in a number of our publications. The corpus consists of 14,785 projects and 352,312,696 LOC.
- [150k Python Dataset](https://www.sri.inf.ethz.ch/py150) - Dataset consisting of 150,000 Python ASTs.
- [150k JavaScript Dataset](https://www.sri.inf.ethz.ch/js150) - Dataset consisting of 150,000 JavaScript files and their parsed ASTs.
- [card2code](https://github.com/deepmind/card2code) - This dataset contains the language to code datasets described in the paper [Latent Predictor Networks for Code Generation](#card2code).
- [NL2Bash](https://github.com/TellinaTool/nl2bash) - This dataset contains a set of ~10,000 bash one-liners collected from websites such as StackOverflow and their English descriptions written by Bash programmers, as described in the [paper](https://arxiv.org/abs/1802.08979).
- [GitHub JavaScript Dump October 2016](https://archive.org/details/javascript-sources-oct2016.sqlite3) - Dataset consisting of 494,352 syntactically-valid JavaScript files obtained from the top ~10000 starred JavaScript repositories on GitHub, with licenses, and parsed ASTs.
- [BigCloneBench](https://jeffsvajlenko.weebly.com/bigcloneeval.html) - Clone detection benchmark of 8 million function clone pairs in the IJaDataset.

## Credits

- A lot of references and articles were taken from [mast-group](https://mast-group.github.io/).
- Inspired by [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning).

## Contributions

See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR: create a [pull request](https://github.com/src-d/awesome-machine-learning-on-source-code/pulls) which is [signed off](https://github.com/src-d/awesome-machine-learning-on-source-code/blob/master/CONTRIBUTING.md#certificate-of-origin).

## License

[![License: CC BY-SA 4.0](badges/License-CC-BY--SA-4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
