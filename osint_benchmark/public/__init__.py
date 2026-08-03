"""Step 4: the public sources that can only be fetched once the entities are known.

The bulk sources in step 1 depend on nothing. These depend on the graph: article text and
Wikidata statements are fetched for bridge entities, and the commercial register is queried
for the firms the reporting actually names -- it holds 2.8M publications, and mirroring it
wholesale would be pointless when only a few hundred companies matter.

Everything here records the revision it read, because these sources are live and a gold
answer taken from one is only correct against a particular version of it.
"""
