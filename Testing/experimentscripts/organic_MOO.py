"""
variable: Seeding rate
objectives: 
    1. net return/yield
    2. Minimizing weed -> one group of all weeds 
                        -> 60-85 points for 80-230 acre fields
                        -> each point is a quarter meter square
                        -> in each square, volume of the weeds is estimated (m3 per m2)
                        -> strong biomass correlation 
                        -> weeds compete with the crops for resources, so more weeds means more weed-seeds which could 
                        take over a field. Theoretically, planting a lot of crop seeds would reduce the weed influence, 
                        however, this has not necessarily been found to be correct.
        Thistle patch growth -> how does seeding rate influence this growth? High seeding rate seems beneficial for the First year,
        but the second year the thistle comes back stronger. But not enough data yet to include in a model.
    3. Rotation of crops to avoid creating optimal ecosystem for bugs and pests 

Organic vs Conventional: maneure spreading is the most expensive part, and more labor.
conventional farming: pesticide control -> ask bruce and paul about data for this: could influence net return since it costs money
plus how beneficial is it actually?

Hannah's bug research: area vs ecological refuge: based on natural features, if there is low yield in an area, 
should we change this to a biodiversity area, since having such an area positively influences yield right outside of it?
--> how can this be changed into an optimization problem.

!!!! Cell size: min 200m because that is the most accurate size for yield monitoring.
NDWI 
"""