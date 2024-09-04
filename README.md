
# DNS Exfiltration Dataset

A large DNS dataset was collected from an active network environment, containing over 50 million DNS queries. 
To ensure privacy, IP addresses were anonymized. The dataset was carefully analyzed to extract features from individual DNS
queries and identify patterns across multiple requests. After processing, the dataset was reduced to around 13 million records,
encompassing both regular DNS traffic and malicious exfiltration attempts using specific tools. To further increase detection difficulty,
a modified version of the dataset with altered request patterns was also created [\[Link\]](https://data.mendeley.com/datasets/c4n7fckkz3/1/files/e7d37892-8b22-4bd2-a5cd-854cc155e4c1).


# Experiment

Before working with the DNS Exfiltration data, I took specific measures to ensure that our model operates on unbiased and balanced datasets.
Our analysis is performed on a balanced DNS Exfiltration dataset. We utilize two types of models: BERT (without preprocessed data, and balanced data)
and Hybrid BERT (balanced data)). The BERT model is trained exclusively on text data to identify DNS Exfiltration attacks, whereas the Hybrid BERT model 
processes both continuous(numeric) and text data.


I tested our BERT(not cleaned, cleaned) and BERT Hybrid models on the cleaned dataset, taking into account the previously mentioned considerations.
Both models achieved 100\% accuracy on the test data, mirroring the same result in training after 10 epochs. This is very optimistic performance which denotes that this 
dataset is not a good dataset to work with for DNS exfiltration detection.
