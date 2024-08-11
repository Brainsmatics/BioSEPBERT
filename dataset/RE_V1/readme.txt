Relation Extraction Document

989 abstracts 10x cross valid.

data format:
{
	'pmid': {
		'text': [[text1], ...], 
		'token': [[token1, token2, ...]],
		'label': [[label1, label2, ...]], 
		'sentences': [[(ss, se, False, os, oe), ...]],
		'sentence_direcitons': [[(ss, se, 0/1/2/3, os, oe), ...]]
	}
}
