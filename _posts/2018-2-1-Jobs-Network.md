---
layout: post
title: Visualizing a jobs network!
---

The idea of jobs being related to each other by related tasks is an interesting one. Made this for an academic think-tank using the US ONET Occupational Database, collected under the US Department of Labour (found here: <https://www.onetonline.org/>)

The visualization can be found here: <https://chowjiahui.github.io/ONETnetwork/>.

Using Python's pandas library, extra data wrangling was done before reading it in under the networkx library as an edge list for a directed graph. 

In Gephi, Rutcherman-Reingold algorithm was first run to 'untangle' the edges into clusters, followed by Force Atlas 2 to for better separation between the clusters.

Something cool I would like to point out is that industries naturally emerge from the clusters, as jobs within the industry are more related by tasks. Other points to explore would be if tasks can be rated more similar to each other by some sort of distance measure in NLP (like cosine similarity).  