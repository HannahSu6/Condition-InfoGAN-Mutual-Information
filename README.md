# Condition-InfoGAN-Mutual-Information

\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{float}

\title{Enhancing CGANs through Integrating Mutual Information}
\author{Qin Su}
\date{April 27, 2024}

\begin{document}

\maketitle

\section{Problem Description}
Generative Adversarial Networks (GANs) utilize two networks: a generator and a discriminator, in a minimax game to learn to generate new data samples that are indistinguishable from the real data \cite{Goodfellow2014}. Conditional GANs (CGANs) extend GANs by conditioning the generation process on additional information, improving control and relevance of generated samples \cite{Mirza2014}. Despite their successes, CGANs face challenges such as mode collapse and instability during training. Information Maximizing GANs (InfoGAN) is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation \cite{Chen2016}. Incorporating mutual information can theoretically ensure that all modes of the data distribution are captured, enhancing diversity and fidelity of the generated samples \cite{Belghazi2018}. This report explores enhancing CGANs by integrating mutual information to improve output diversity and quality.

\section{Background and Motivation}
GANs introduced the principle of adversarial training, involving a dynamic where two models, the generator (G) and the discriminator (D), are pitted against each other in a game-theoretic framework. Specifically, the discriminator's job is to accurately differentiate between real and fake data, whereas the generator strives to produce data so convincing that it becomes indistinguishable from real data. This interaction is succinctly captured by the GAN objective function:

\begin{equation}
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))]
\end{equation}

where \( p_{\text{data}} \) is the real data distribution, and \( p_{\mathbf{z}} \) is a prior over input noise variables. This groundbreaking strategy not only encourages the generation of new data that mirrors the original training dataset but also enhances its quality through continuous iterative training \cite{Goodfellow2014}.

Building on the GAN model, CGANs were developed to generate data conditioned on additional information such as labels or features. This advancement enables targeted data generation, crucial for tasks requiring specified outcomes. The CGAN framework modifies the GAN objective to incorporate conditionality:

\begin{equation}
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x}|y)} [\log D(\mathbf{x}|y)] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})} [\log(1 - D(G(\mathbf{z}|y)))]
\end{equation}

Here, \( y \) represents the conditioning variable, allowing for more directed and versatile data generation. This extension allows CGANs to direct the data generation process, highlighting the importance of efficiently processing conditional inputs, notably through convolutional neural networks \cite{Mirza2014}. CGANs have shown promise in tasks like image synthesis, where the model conditions on labels or attributes, as shown in Fig. \ref{fig:cgan}.

\begin{figure}[H]
\centering
\includegraphics[width=0.5\textwidth]{fig1.png}
\caption{Generated MNIST digits, each row conditioned on one label \cite{Mirza2014}.}
\label{fig:cgan}
\end{figure}

However, the generation quality often varies significantly depending on the complexity of the conditioning data and the architecture used. Mutual information, a measure of the amount of information one random variable contains about another, can be leveraged to optimize the information flow between the conditioning labels and the generated outputs, as shown in Fig. \ref{fig:infogan}.

\begin{figure}[H]
\centering
\includegraphics[width=0.5\textwidth]{fig2.png}
\caption{Manipulating Latent Codes on MNIST: Each figure demonstrates variations in a specific latent code from left to right, with other codes and noise held constant \cite{Chen2016}.}
\label{fig:infogan}
\end{figure}

Different rows showcase various random samples of these fixed elements:
\begin{itemize}
    \item (a): Each column displays five samples from a single category in \( c1 \), with a row depicting generated images across 10 categories in \( c1 \). Here, \( c1 \) primarily correlates with distinct digit types.
    \item (b): Variation in \( c1 \) on a GAN without information regularization leads to non-interpretable outcomes.
    \item (c): A low value of \( c2 \) causes the digits to lean to the left, whereas a high value of \( c2 \) makes them tilt to the right.
    \item (d): \( c3 \) adjusts the digit width smoothly. 
\end{itemize}

Unlike CGANs, the generated digits are organized with labels that match the conditions. This project is motivated by the hypothesis that enhancing CGANs with mutual information can lead to more stable and high-quality generation, particularly in multimodal data settings. Simultaneously, integrating CGANs with InfoGAN could allow the original InfoGAN to generate ordered digits not by categorical code but through conditional labels.

\section{Methods}
The approach involves two key methodological advancements:

\subsection{Architectural Enhancements}
Elements from Information Maximizing GANs (InfoGAN) are integrated into the CGAN architecture. This involves adapting the generator of the CGAN to maximize mutual information between a subset of latent codes and the generated outputs. Therefore, we propose to solve the following information-regularized minimax game \cite{Chen2016} \cite{Belghazi2018}:

\begin{equation}
\min_G \max_D V(D, G) = V(D, G) + \lambda I(c; G(\mathbf{z}|y, c))
\end{equation}

Utilizing the architecture of InfoGAN, which reveals key structured semantic features from the latent space, is designed to enhance both the control and diversity of the data generated.

\subsection{Mutual Information Maximization}
In practice, the mutual information term \( I(c; G(\mathbf{z}|y, c)) \) is hard to maximize directly as it requires access to the posterior \( P(c|\mathbf{x}) \). Fortunately, we can obtain a lower bound of it by defining an auxiliary distribution \( Q(c|\mathbf{x}) \) to approximate \( P(c|\mathbf{x}) \) and we can define a variational lower bound, \( L(G|Q) \), of the mutual information \( I(c; G(\mathbf{z}|y, c)) \) \cite{Chen2016}.

Hence, CGANs with mutual information maximization (C-InfoGAN) is defined as the following minimax game with a variational regularization of mutual information and a hyperparameter \( \lambda \) \cite{Chen2016}:

\begin{equation}
\min_{G, Q} \max_D V(D, G) = V(D, G) + \lambda L(G|Q)
\end{equation}

\section{Experimental Settings and Results}
\subsection{Datasets}
The study utilized the MNIST dataset for digit generation.

\subsection{Training Details}
Since GAN is known to be difficult to train, we design our experiments based on existing techniques introduced by DC-GAN \cite{Radford2015}, which are enough to stabilize C-InfoGAN training and we did not have to introduce new tricks. The training configuration included the use of the Adam optimizer, with specific batch sizes and learning rates adjusted over the course of training \cite{Salimans2016}.

The main aim of the experiments is to see if C-InfoGAN can not only make the generated images more diverse and clearer, but also can learn disentangled and interpretable representations. This involves using the generator to change one latent factor at a time to check if this change affects only one aspect of the image.

To disentangle digit shape from styles on MNIST, the experiments choose class labels encoded as one-hot vectors and two continuous codes that can capture variations that are continuous in nature: \( c1, c2 \sim \text{Uniform} (-1, 1) \). In Fig. \ref{fig:mnist}, we show that each row conditioned on one label in (a), each column conditioned on one label in (b) and each figure demonstrates variations in a specific latent code from left to right, with other codes and noise held constant in (c) and (d).

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{fig3.png}
\caption{Manipulating Label and Latent Codes on MNIST.}
\label{fig:mnist}
\end{figure}

The second goal is to confirm that C-InfoGAN could allow the original InfoGAN to generate ordered digits not by categorical code but through conditional labels, as shown in Fig. \ref{fig


[6] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved 
techniques for training GANs,” in Advances in Neural Information Processing Systems, 2016. 




