# Awesome Machine Learning On Source Code [![Awesome Machine Learning On Source Code](https://awesome.re/badge.svg)](https://github.com/src-d/awesome-machine-learning-on-source-code)
A curated list of awesome machine learning frameworks and algorithms that work on top of source code. Inspired by [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning).

## Contents
<!-- MarkdownTOC depth=4 -->
- [Digests](#digests)
- [Conferences](#conferences)
- [Papers](#papers)
    - [Program Synthesis and Induction](#program-synthesis-induction)
    - [Source Code Analysis and Language modeling](#code-analysis)
    - [Neural Network Architectures and Algorithms](#rnn-algo)
    - [Program Translation](#program-translation)
    - [Code Suggestion and Completion](#code-suggestion-completion)
    - [Program Repair and Bug Detection](#program-repair-bug-detection)
    - [APIs and Code Mining](#api-mining)
    - [Code Optimization](#code-optimization)
    - [Topic Modeling](#topic-modeling)
    - [Code Summarization](#code-summarization)
    - [Clone Detection](#clone-detection)
    - [Differentiable Interpreters](#differentiable-interpreters)
    - [Related research](#related-research)<details><summary>(links require "Related research" spoiler to be open)</summary>
        - [Binary Data Modelling](#binary-data-modelling)
        - [Soft Clustering Using T-mixture Models](#t-mixture-models)
        </details>

- [Posts](#posts)
- [Talks](#talks)
- [Software](#software)
    - [Machine Learning](#software-ML)
    - [Utilities](#software-utilities)
- [Datasets](#datasets)
- [Credits](#credits)
- [Contributions](#contributions)
- [License](#license)

<!-- /MarkdownTOC -->

<a name="digests"></a>
## Digests
* [Learning from "Big Code"](http://learnbigcode.github.io)
* [A Survey of Machine Learning for Big Code and Naturalness](https://ml4code.github.io/)

<a name="conferences"></a>
## Conferences
* [Workshop on NLP for Software Engineering](https://nl4se.github.io/)
* [SysML](http://www.sysml.cc/)
    * [Talks](https://www.youtube.com/channel/UChutDKIa-AYyAmbT45s991g/)
* [Mining Software Repositories](http://www.msrconf.org/)
* [AIFORSE](aiforse.org)
* [source{d} tech talks](https://talks.sourced.tech/machine-learning-2017/)
    * [Talks](https://www.youtube.com/playlist?list=PL5Ld68ole7j3iQFUSB3fR9122dHCUWXsy)
* [NIPS Neural Abstract Machines and Program Induction workshop](ucmlr.github.io/nampi)
    * [Talks](https://www.youtube.com/playlist?list=PLzTDea_cM27LVPSTdK9RypSyqBHZWPywt)


<a name="papers"></a>
## Papers

<a name="program-synthesis-induction"></a>
#### Program Synthesis and Induction
* [NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System](https://arxiv.org/abs/1802.08979v2) - Xi Victoria Lin, Chenglong Wang, Luke Zettlemoyer, Michael D. Ernst, 2018. 12p
* [Recent Advances in Neural Program Synthesis](https://arxiv.org/abs/1802.02353v1) - Neel Kant, 2018. 18p
* [Neural Sketch Learning for Conditional Program Generation](https://arxiv.org/abs/1703.05698) - Vijayaraghavan Murali, Letao Qi, Swarat Chaudhuri, Chris Jermaine, 2018. 16p
* [Neural Program Search: Solving Programming Tasks from Description and Examples](https://arxiv.org/abs/1802.04335v1) - Illia Polosukhin, Alexander Skidanov, 2018. 11p
* [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526v1) - Daniel A. Abolafia, Mohammad Norouzi, Quoc V. Le, 2018. 16p
* [Towards Synthesizing Complex Programs from Input-Output Examples](https://arxiv.org/abs/1706.01284v3) - Xinyun Chen, Chang Liu, Dawn Song, 2018. 31p
* [SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning](https://arxiv.org/abs/1711.04436v1) - Xiaojun Xu, Chang Liu, Dawn Song, 2017. 13p
* [Learning to Select Examples for Program Synthesis](https://arxiv.org/abs/1711.03243v1) - Yewen Pu, Zachery Miranda, Armando Solar-Lezama, Leslie Pack Kaelbling, 2017. 12p
* [Neural Program Meta-Induction](https://arxiv.org/abs/1710.04157v1) - Jacob Devlin, Rudy Bunel, Rishabh Singh, Matthew Hausknecht, Pushmeet Kohli, 2017. 10p
* [Glass-Box Program Synthesis: A Machine Learning Approach](https://arxiv.org/abs/1709.08669v1) - Konstantina Christakopoulou, Adam Tauman Kalai, 2017. 8p
* [Learning to Infer Graphics Programs from Hand-Drawn Images](https://arxiv.org/abs/1707.09627v4) - Kevin Ellis, Daniel Ritchie, Armando Solar-Lezama, Joshua B. Tenenbaum, 2017. 14p
* [Neural Attribute Machines for Program Generation](https://arxiv.org/abs/1705.09231v2) - Matthew Amodio, Swarat Chaudhuri, Thomas Reps, 2017. 10p
* [Abstract Syntax Networks for Code Generation and Semantic Parsing](https://arxiv.org/abs/1704.07535v1) - Maxim Rabinovich, Mitchell Stern, Dan Klein, 2017. 11p
* [Making Neural Programming Architectures Generalize via Recursion](https://arxiv.org/pdf/1704.06611v1.pdf) - Jonathon Cai, Richard Shin, Dawn Song, 2017. 20p
* [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696v1) - Pengcheng Yin, Graham Neubig, 2017. 14p
* [Program Synthesis from Natural Language Using Recurrent Neural Networks](https://homes.cs.washington.edu/~mernst/pubs/nl-command-tr170301.pdf) - Xi Victoria Lin, Chenglong Wang, Deric Pang, Kevin Vu, Luke Zettlemoyer, Michael Ernst, 2017. 12p
* [RobustFill: Neural Program Learning under Noisy I/O](https://arxiv.org/abs/1703.07469v1) - Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, Pushmeet Kohli, 2017. 18p
* [Lifelong Perceptual Programming By Example](https://openreview.net/pdf?id=HJStZKqel) - Gaunt, Alexander L., Marc Brockschmidt, Nate Kushman, and Daniel Tarlow, 2017. 11p
* [Neural Programming by Example](https://arxiv.org/abs/1703.04990v1) - Chengxun Shu, Hongyu Zhang, 2017. 7p
* [DeepCoder: Learning to Write Programs](http://arxiv.org/abs/1611.01989) - Balog, Matej, Alexander L. Gaunt, Marc Brockschmidt, Sebastian Nowozin, and Daniel Tarlow, 2017. 21p
* [A Differentiable Approach to Inductive Logic Programming](https://uclmr.github.io/nampi/extended_abstracts/yang.pdf) - Yang, Fan, Zhilin Yang, and William W. Cohen, 2017. 10p
* [Latent Attention For If-Then Program Synthesis](https://arxiv.org/abs/1611.01867v1) - Xinyun Chen, Chang Liu, Richard Shin, Dawn Song, Mingcheng Chen, 2016. 12p
* [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744) - Wang Ling, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočiský, Andrew Senior, Fumin Wang, Phil Blunsom, 2016. 11p
* [Meta-Interpretive Learning of Efficient Logic Programs](https://uclmr.github.io/nampi/extended_abstracts/cropper.pdf) - Cropper, Andrew, and Stephen H. Muggleton, 2016. 2p
* [Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision (Short Version)](http://arxiv.org/abs/1612.01197) - Liang, Chen, Jonathan Berant, Quoc Le, Kenneth D. Forbus, and Ni Lao, 2016. 6p
* [Programs as Black-Box Explanations](http://arxiv.org/abs/1611.07579) - Singh, Sameer, Marco Tulio Ribeiro, and Carlos Guestrin, 2016. 5p
* [Structured Generative Models of Natural Source Code](https://arxiv.org/abs/1401.0514) - Chris J. Maddison, Daniel Tarlow, 2014. 14p

<a name="code-analysis"></a>
#### Source Code Analysis and Language modeling
* [A Survey of Machine Learning for Big Code and Naturalness](https://arxiv.org/abs/1709.06182v1) - Miltiadis Allamanis, Earl T. Barr, Premkumar Devanbu, Charles Sutton, 2017. 36p
* [Learning to Represent Programs with Graphs](https://arxiv.org/abs/1711.00740v1) - Miltiadis Allamanis, Marc Brockschmidt, Mahmoud Khademi, 2017. 16p
* [A deep language model for software code](https://arxiv.org/abs/1608.02715v1) - Hoa Khanh Dam, Truyen Tran, Trang Pham, 2016. 4p
* [Suggesting Accurate Method and Class Names](http://homepages.inf.ed.ac.uk/csutton/publications/accurate-method-and-class.pdf) - Miltiadis Allamanis, Earl T. Barr, Christian Bird, Charles Sutton, 2015. 12p
* [Mining Source Code Repositories at Massive Scale using Language Modeling](http://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf) - Miltiadis Allamanis, Charles Sutton, 2013. 10p

<a name="rnn-algo"></a>
#### Neural Network Architectures and Algorithms
* [Learning Approximate Inference Networks for Structured Prediction](https://arxiv.org/abs/1803.03376v1) - Lifu Tu, Kevin Gimpel, 2018. 15p
* [Learning Deep Generative Models of Graphs](https://arxiv.org/abs/1803.03324v1.pdf) - Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, Peter Battaglia, 2018. 21p
* [Cross-Language Learning for Program Classification using Bilateral Tree-Based Convolutional Neural Networks](https://arxiv.org/abs/1710.06159v2) - Nghi D. Q. Bui, Lingxiao Jiang, Yijun Yu, 2017. 4p
* [Syntax-Directed Variational Autoencoder for Structured Data](https://openreview.net/pdf?id=SyqShMZRb) - Hanjun Dai, Yingtao Tian, Bo Dai, Steven Skiena, Le Song, 2018. 17p
* [Divide and Conquer with Neural Networks](http://arxiv.org/abs/1611.02401) - Nowak, Alex, and Joan Bruna, 2017. 19p
* [Learning Efficient Algorithms with Hierarchical Attentive Memory](http://arxiv.org/abs/1602.03218) - Andrychowicz, Marcin, and Karol Kurach, 2016. 10p
* [Learning Operations on a Stack with Neural Turing Machines](http://arxiv.org/abs/1612.00827) - Deleu, Tristan, and Joseph Dureau, 2016. 6p
* [Probabilistic Neural Programs](http://arxiv.org/abs/1612.00712) - Murray, Kenton W., and Jayant Krishnamurthy, 2016. 5p
* [Learning Latent Multiscale Structure Using Recurrent Neural Networks](https://uclmr.github.io/nampi/extended_abstracts/chung.pdf) - Chung, Junyoung, Sungjin Ahn, and Yoshua Bengio, 2016. 3p
* [Neural Programmer: Inducing Latent Programs with Gradient Descent](http://arxiv.org/abs/1511.04834) - Neelakantan, Arvind, Quoc V. Le, and Ilya Sutskever, 2016. 18p
* [Neural Programmer-Interpreters](http://arxiv.org/abs/1511.06279) - Reed, Scott, and Nando de Freitas, 2016. 13p
* [Neural GPUs Learn Algorithms](http://arxiv.org/abs/1511.08228) - Kaiser, Łukasz, and Ilya Sutskever, 2016. 9p
* [Neural Random-Access Machines](https://arxiv.org/abs/1511.06392v3) - Karol Kurach, Marcin Andrychowicz, Ilya Sutskever, 2016. 17p
* [Learning to Execute](https://arxiv.org/abs/1410.4615v3) - Wojciech Zaremba, Ilya Sutskever, 2015. 25p
* [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/abs/1503.01007) - Joulin, Armand, and Tomas Mikolov, 2015. 10p
* [Neural Turing Machines](http://arxiv.org/abs/1410.5401) - Graves, Alex, Greg Wayne, and Ivo Danihelka, 2014. 26p
* [From Machine Learning to Machine Reasoning](http://arxiv.org/abs/1102.1808) - Bottou, Leon, 2011. 15p

<a name="program-translation"></a>
#### Program Translation
* [Hierarchical Learning of Cross-Language Mappings through Distributed Vector Representations for Code](https://arxiv.org/abs/1803.04715v1) - Nghi D. Q. Bui, Lingxiao Jiang, 2018. 4p
* [Tree-to-tree Neural Networks for Program Translation](https://arxiv.org/abs/1802.03691v1) - Xinyun Chen, Chang Liu, Dawn Song, 2018. 14p
* [Code Attention: Translating Code to Comments by Exploiting Domain Features](https://arxiv.org/abs/1709.07642v2) - Wenhao Zheng, Hong-Yu Zhou, Ming Li, Jianxin Wu, 2017. 12p
* [Automatically Generating Commit Messages from Diffs using Neural Machine Translation](https://arxiv.org/abs/1708.09492v1) - Siyuan Jiang, Ameer Armaly, Collin McMillan, 2017. 12p
* [A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation](https://arxiv.org/abs/1707.02275v1) - Antonio Valerio Miceli Barone, Rico Sennrich, 2017. 5p
* [A Neural Architecture for Generating Natural Language Descriptions from Source Code Changes](https://arxiv.org/abs/1704.04856v1) - Pablo Loyola, Edison Marrese-Taylor, Yutaka Matsuo, 2017. 6p

<a name="code-suggestion-completion"></a>
#### Code Suggestion and Completion
* [Code Completion with Neural Attention and Pointer Networks](https://arxiv.org/abs/1711.09573v1) - Jian Li, Yue Wang, Irwin King, Michael R. Lyu, 2017. 8p
* [Learning Python Code Suggestion with a Sparse Pointer Network](https://arxiv.org/abs/1611.08307) - Avishkar Bhoopchand, Tim Rocktäschel, Earl Barr, Sebastian Riedel, 2016. 11p
* [Code Completion with Statistical Language Models](http://www.srl.inf.ethz.ch/papers/pldi14-statistical.pdf) - Veselin Raychev, Martin Vechev, Eran Yahav, 2014. 10p

<a name="program-repair-bug-detection"></a>
#### Program Repair and Bug Detection
* [Dynamic Neural Program Embedding for Program Repair](https://arxiv.org/abs/1711.07163v2) - Ke Wang, Rishabh Singh, Zhendong Su, 2018. 11p
* [To Type or Not to Type: Quantifying Detectable Bugs in JavaScript](http://earlbarr.com/publications/typestudy.pdf) - Zheng Gao, Christian Bird, Earl Barr, 2017. 12p
* [Semantic Code Repair using Neuro-Symbolic Transformation Networks](https://arxiv.org/abs/1710.11054v1) - Jacob Devlin, Jonathan Uesato, Rishabh Singh, Pushmeet Kohli, 2017. 11p
* [Automated Identification of Security Issues from Commit Messages and Bug Reports](http://asankhaya.github.io/pdf/automated-identification-of-security-issues-from-commit-messages-and-bug-reports.pdf) - Yaqin Zhou and Asankhaya Sharma, 2017. 6p
* [SmartPaste: Learning to Adapt Source Code](https://arxiv.org/abs/1705.07867) - Miltiadis Allamanis, Marc Brockschmidt, 2017. 31p
* [End-to-End Prediction of Buffer Overruns from Raw Source Code via Neural Memory Networks](https://arxiv.org/abs/1703.02458v1) - Min-je Choi, Sehun Jeong, Hakjoo Oh, Jaegul Choo, 2017. 7p
* [Tailored Mutants Fit Bugs Better](https://arxiv.org/abs/1611.02516) - Miltiadis Allamanis, Earl T. Barr, René Just, Charles Sutton, 2016. 11p

<a name="api-mining"></a>
#### APIs and Code Mining
* [DeepAM: Migrate APIs with Multi-modal Sequence to Sequence Learning](https://arxiv.org/abs/1704.07734v1) - Xiaodong Gu, Hongyu Zhang, Dongmei Zhang, Sunghun Kim, 2017. 7p
* [Deep API Learning](https://arxiv.org/abs/1605.08535v3) - Xiaodong Gu, Hongyu Zhang, Dongmei Zhang, Sunghun Kim, 2017. 12p
* [API usage pattern recommendation for software development](http://www.sciencedirect.com/science/article/pii/S0164121216301200) - Haoran Niu, Iman Keivanloo, Ying Zou, 2017. 12p
* [Exploring API Embedding for API Usages and Applications](http://home.eng.iastate.edu/~trong/projects/jv2cs/) - Nguyen, Nguyen, Phan and Nguyen, 2017. 11p
* [Parameter-Free Probabilistic API Mining across GitHub](http://homepages.inf.ed.ac.uk/csutton/publications/fse2016.pdf) - Jaroslav Fowkes, Charles Sutton, 2016. 12p
* [A Subsequence Interleaving Model for Sequential Pattern Mining](http://homepages.inf.ed.ac.uk/csutton/publications/kdd2016-subsequence-interleaving.pdf) - Jaroslav Fowkes, Charles Sutton, 2016. 10p
* [Lean GHTorrent: GitHub data on demand](https://bvasiles.github.io/papers/lean-ghtorrent.pdf) - Georgios Gousios, Bogdan Vasilescu, Alexander Serebrenik, Andy Zaidman, 2014. 4p
* [Mining idioms from source code](http://homepages.inf.ed.ac.uk/csutton/publications/idioms.pdf) - Miltiadis Allamanis, Charles Sutton, 2014. 12p
* [The GHTorent Dataset and Tool Suite](http://www.gousios.gr/pub/ghtorrent-dataset-toolsuite.pdf) - Georgios Gousios, 2013. 4p

<a name="code-optimization"></a>
#### Code Optimization
* [A Survey on Compiler Autotuning using Machine Learning](https://arxiv.org/abs/1801.04405v2.pdf) - Amir H. Ashouri, William Killian, John Cavazos, Gianluca Palermo, Cristina Silvano,2018. 42p
* [Learning Memory Access Patterns](https://arxiv.org/abs/1803.02329v1.pdf) - Milad Hashemi, Kevin Swersky, Jamie A. Smith, Grant Ayers, Heiner Litz, Jichuan Chang, Christos Kozyrakis, Parthasarathy Ranganathan, 2018. 15p
* [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208v2) - Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, Neoklis Polyzotis, 2017. 27p
* [Learning to superoptimize programs](https://arxiv.org/abs/1611.01787v3) - Rudy Bunel, Alban Desmaison, M. Pawan Kumar, Philip H.S. Torr, Pushmeet Kohlim 2017. 14p
* [Neural Nets Can Learn Function Type Signatures From Binaries](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-chua.pdf) - Zheng Leong Chua, Shiqi Shen, Prateek Saxena, and Zhenkai Liang, 2017. 18p
* [Adaptive Neural Compilation](https://arxiv.org/abs/1605.07969v2) - Rudy Bunel, Alban Desmaison, Pushmeet Kohli, Philip H.S. Torr, M. Pawan Kumar, 2016. 25p
* [Learning to Superoptimize Programs - Workshop Version](http://arxiv.org/abs/1612.01094) - Bunel, Rudy, Alban Desmaison, M. Pawan Kumar, Philip H. S. Torr, and Pushmeet Kohli, 2016. 10p

<a name="topic-modeling"></a>
#### Topic Modeling
* [Topic modeling of public repositories at scale using names in source code](https://arxiv.org/abs/1704.00135) - Vadim Markovtsev, Eiso Kant, 2017. 11p
* [Why, When, and What: Analyzing Stack Overflow Questions by Topic, Type, and Code](http://homepages.inf.ed.ac.uk/csutton/publications/msrCh2013.pdf) - Miltiadis Allamanis, Charles Sutton, 2013. 4p
* [Semantic clustering: Identifying topics in source code](https://pdfs.semanticscholar.org/c9ba/722322912419e59ea251c22b437d251f1644.pdf) - Adrian Kuhn, Stéphane Ducasse, Tudor Girba, 2007. 30p

<a name="code-summarization"></a>
#### Code Summarization
* [A Convolutional Attention Network for Extreme Summarization of Source Code](http://arxiv.org/abs/1602.03001) - Miltiadis Allamanis, Hao Peng, Charles Sutton, 2016. 11p
* [TASSAL: Autofolding for Source Code Summarization](http://homepages.inf.ed.ac.uk/csutton/publications/icse2016-demo.pdf) - Jaroslav Fowkes, Pankajan Chanthirasegaran, Razvan Ranca, Miltiadis Allamanis, Mirella Lapata, Charles Sutton, 2016. 4p
* [Summarizing Source Code using a Neural Attention Model](https://github.com/sriniiyer/codenn/blob/master/summarizing_source_code.pdf) - Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, Luke Zettlemoyer, 2016. 11p

<a name="clone-detection"></a>
#### Clone Detection
* [DéjàVu: a map of code duplicates on GitHub](http://janvitek.org/pubs/oopsla17b.pdf) - Cristina V. Lopes,  	Petr Maj, Pedro Martins, Vaibhav Saini, Di Yang, Jakub Zitny, Hitesh Sajnani, Jan Vitek, 2017. 28p
* [Some from Here, Some from There: Cross-project Code Reuse in GitHub](http://web.cs.ucdavis.edu/~filkov/papers/clones.pdf) - Mohammad Gharehyazie, Baishakhi Ray, Vladimir Filkov, 2017. 11p
* [Deep Learning Code Fragments for Code Clone Detection](http://www.cs.wm.edu/~denys/pubs/ASE'16-DeepLearningClones.pdf) - Martin White, Michele Tufano, Christopher Vendome, and Denys Poshyvanyk, 2016. 12p
* [A study of repetitiveness of code changes in software evolution](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?referer=https://scholar.google.com/&httpsredir=1&article=1016&context=cs_conf) - HA Nguyen, AT Nguyen, TT Nguyen, TN Nguyen, H Rajan, 2013. 11p

<a name="differentiable-interpreters"></a>
#### Differentiable Interpreters
* [Improving the Universality and Learnability of Neural Programmer-Interpreters with Combinator Abstraction](https://arxiv.org/abs/1802.02696v1) - Da Xiao, Jo-Yu Liao, Xingyuan Yuan, 2018. 16p
* [Differentiable Programs with Neural Libraries](https://arxiv.org/abs/1611.02109v2) - Alexander L. Gaunt, Marc Brockschmidt, Nate Kushman, Daniel Tarlow, 2017. 10p
* [Differentiable Functional Program Interpreters](https://arxiv.org/abs/1611.01988v2) - John K. Feser, Marc Brockschmidt, Alexander L. Gaunt, Daniel Tarlow, 2017. 15p
* [Programming with a Differentiable Forth Interpreter](http://arxiv.org/abs/1605.06640) - Bošnjak, Matko, Tim Rocktäschel, Jason Naradowsky, and Sebastian Riedel, 2017. 18p
* [Neural Functional Programming](http://arxiv.org/abs/1611.01988) - Feser, John K., Marc Brockschmidt, Alexander L. Gaunt, and Daniel Tarlow, 2017. 15p
* [TerpreT: A Probabilistic Programming Language for Program Induction](http://arxiv.org/abs/1612.00817) - Gaunt, Alexander L., Marc Brockschmidt, Rishabh Singh, Nate Kushman, Pushmeet Kohli, Jonathan Taylor, and Daniel Tarlow, 2016. 7p


<a name="related-research"></a>
<details>
<summary>Related research</summary>

<a name="binary-data-modelling"></a>
#### Binary Data Modelling
* [Clustering Binary Data with Bernoulli Mixture Models](http://nsgrantham.com/documents/clustering-binary-data.pdf) - Neal S. Grantham
* [A Family of Blockwise One-Factor Distributions for Modelling High-Dimensional Binary Data](https://arxiv.org/pdf/1511.01343.pdf) - Matthieu Marbac and Mohammed Sedki
* [BayesBinMix: an R Package for Model Based Clustering of Multivariate Binary Data](https://arxiv.org/pdf/1609.06960.pdf) - Panagiotis Papastamoulis and Magnus Rattray

<a name="t-mixture-models"></a>
#### Soft Clustering Using T-mixture Models
* [Robust mixture modelling using the t distribution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.7334&rep=rep1&type=pdf) - D. PEEL and G. J. MCLACHLAN
* [Robust mixture modeling using the skew t distribution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1030.9865&rep=rep1&type=pdf) - Tsung I. Lin, Jack C. Lee and Wan J. Hsieh
</details>

<a name="posts"></a>
## Posts
* [Sequence Intent Classification Using Hierarchical Attention Networks](https://www.microsoft.com/developerblog/2018/03/06/sequence-intent-classification/)
* [Syntax-Directed Variational  Autoencoder for Structured Data](https://mlatgt.blog/2018/02/08/syntax-directed-variational-autoencoder-for-structured-data/)
* [Weighted MinHash on GPU helps to find duplicate GitHub repositories.](https://blog.sourced.tech//post/minhashcuda/)
* [Source Code Identifier Embeddings](https://blog.sourced.tech/post/id2vec/)
* [Using recurrent neural networks to predict next tokens in the java solutions](http://near.ai/articles/2017-06-01-Code-Completion-Demo/)
* [The half-life of code & the ship of Theseus](https://erikbern.com/2016/12/05/the-half-life-of-code.html)
* [The eigenvector of "Why we moved from language X to language Y"](https://erikbern.com/2017/03/15/the-eigenvector-of-why-we-moved-from-language-x-to-language-y.html)
* [Analyzing Github, How Developers Change Programming Languages Over Time](https://blog.sourced.tech/post/language_migrations/)

<a name="talks"></a>
## Talks
* [Topic Modeling of GitHub Repositories](https://blog.sourced.tech//post/github_topic_modeling/)
* [Similarity of GitHub Repositories by Source Code Identifiers](http://vmarkovtsev.github.io/techtalks-2017-moscow/)
* [Using deep RNN to model source code](http://vmarkovtsev.github.io/re-work-2016-london/)
* [Source code abstracts classification using CNN (1)](http://vmarkovtsev.github.io/re-work-2016-berlin/)
* [Source code abstracts classification using CNN (2)](http://vmarkovtsev.github.io/data-natives-2016/)
* [Source code abstracts classification using CNN (3)](http://vmarkovtsev.github.io/slush-2016/)
* [Embedding the GitHub contribution graph](https://egorbu.github.io/techtalks-2017-moscow)
* [Measuring code sentiment in a Git repository](http://vmarkovtsev.github.io/gophercon-2018-moscow/)

<a name="software"></a>
## Software

<a name="software-ML"></a>
#### Machine Learning
* [Differentiable Neural Computer (DNC)](https://github.com/deepmind/dnc) - TensorFlow implementation of the Differentiable Neural Computer.
* [sourced.ml](https://github.com/src-d/ml) - Abstracts feature extraction from source code syntax trees and working with ML models.
* [vecino](https://github.com/src-d/vecino) - Finds similar Git repositories.
* [apollo](https://github.com/src-d/apollo) - Source code deduplication as scale, research.
* [gemini](https://github.com/src-d/gemini) - Source code deduplication as scale, production.
* [enry](https://github.com/src-d/enry) - Insanely fast file based programming language detector.
* [hercules](https://github.com/src-d/hercules) - Git repository mining framework with batteries on top of go-git.
* [Code Neuron](https://github.com/vmarkovtsev/codeneuron) - Recurrent neural network to detect code blocks in natural language text.
* [Naturalize](https://github.com/mast-group/naturalize) - Language agnostic framework for learning coding conventions from a codebase and then expoiting this information for suggesting better identifier names and formatting changes in the code. 
* [Extreme Source Code Summarization](https://github.com/mast-group/convolutional-attention ) - Convolutional attention neural network that learns to summarize source code into a short method name-like summary by just looking at the source code tokens. 
* [Summarizing Source Code using a Neural Attention Model](https://github.com/sriniiyer/codenn) - CODE-NN, uses LSTM networks with attention to produce sentences that describe C# code snippets and SQL queries from StackOverflow. Torch over C#/SQL
* [Probabilistic API Miner](https://github.com/mast-group/api-mining) - Near parameter-free probabilistic algorithm for mining the most interesting API patterns from a list of API call sequences. 
* [Interesting Sequence Miner](https://github.com/mast-group/sequence-mining) - Novel algorithm that mines the most interesting sequences under a probabilistic model. It is able to efficiently infer interesting sequences directly from the database. 
* [TASSAL](https://github.com/mast-group/tassal) - Tool for the automatic summarization of source code using autofolding. Autofolding automatically creates a summary of a source code file by folding non-essential code and comment blocks.
* [JNice2Predict](http://www.nice2predict.org/) - Efficient and scalable open-source framework for structured prediction, enabling one to build new statistical engines more quickly.

<a name="software-utilities"></a>
#### Utilities
* [go-git](https://github.com/src-d/go-git) - Highly extensible Git implementation in pure Go which is friendly to data mining.
* [bblfsh](https://github.com/bblfsh) - Self-hosted server for source code parsing.
* [engine](https://github.com/src-d/engine) - Scalable and distributed data retrieval pipeline for source code.
* [minhashcuda](https://github.com/src-d/minhashcuda) - Weighted MinHash implementation on CUDA to efficiently find duplicates.
* [kmcuda](https://github.com/src-d/kmcuda) - k-means on CUDA to cluster and to search for nearest neighbors in dense space.
* [wmd-relax](https://github.com/src-d/wmd-relax) - Python package which finds nearest neighbors at Word Mover's Distance.


<a name="datasets"></a>
#### Datasets
* [Public Git Archive](https://github.com/src-d/datasets/tree/master/PublicGitArchive) - 3 TB of Git repositories from GitHub.
* [GitHub repositories - languages distribution](https://data.world/source-d/github-repositories-languages-distribution) - Programming languages distribution in 14,000,000 repositories on GitHub (October 2016).
* [452M commits on GitHub](https://data.world/vmarkovtsev/452-m-commits-on-github) - ≈ 452M commits' metadata from 16M repositories on GitHub (October 2016).
* [GitHub readme files](https://data.world/vmarkovtsev/github-readme-files) - Readme files of all GitHub repositories (16M) (October 2016).
* [from language X to Y](https://data.world/vmarkovtsev/from-language-x-to-y) - Cache file Erik Bernhardsson collected for his awesome blog post.
* [GitHub word2vec 120k](https://data.world/vmarkovtsev/github-word-2-vec-120-k) - Sequences of identifiers extracted from top starred 120,000 GitHub repos.
* [GitHub Source Code Names](https://data.world/vmarkovtsev/github-source-code-names) - Names in source code extracted from 13M GitHub repositories, not people.
* [GitHub duplicate repositories](https://data.world/vmarkovtsev/github-duplicate-repositories) - GitHub repositories not marked as forks but very similar to each other.
* [GitHub lng keyword frequencies](https://data.world/vmarkovtsev/github-lng-keyword-frequencies) - Programming language keyword frequency extracted from 16M GitHub repositories.
* [GitHub Java Corpus](http://groups.inf.ed.ac.uk/cup/javaGithub/ ) - GitHub Java corpus is a set of Java projects collected from GitHub that we have used in a number of our publications. The corpus consists of 14,785 projects and 352,312,696 LOC.
* [150k Python Dataset](http://www.srl.inf.ethz.ch/py150.php) - Dataset consisting of 150'000 Python ASTs.
* [150k JavaScript Dataset](http://www.srl.inf.ethz.ch/js150.php) - Dataset consisting of 150'000 JavaScript files and their parsed ASTs.
* [card2code](https://github.com/deepmind/card2code) - This dataset contains the language to code datasets described in the paper [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744).
* [NL2Bash](https://github.com/TellinaTool/nl2bash) - This dataset contains a set of ~10,000 bash one-liners collected from websites such as StackOverflow and their English descriptions written by Bash programmers, as described in the [paper](https://arxiv.org/abs/1802.08979).

<a name="credits"></a>
## Credits

* A lot of references and articles were taken from [mast-group](https://mast-group.github.io/)

<a name="contributions"></a>
## Contributions

See [CONTRIBUTING.md](CONTRIBUTING.md).

<a name="license"></a>
## License

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
