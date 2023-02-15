# SEACARTO
Gravity model of maritime trade featuring narratives of antagonism\
To run the model follow the steps:<br/>
1. Clone the git repository on your local machine<br/>
2. Download the following data:<br/>
  2.1 CEPPI Gravity modelling dataset as csv and store it into the folder /data/gravity_model (http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8)<br/>
  2.2 BACI (HS92) data as csv and store it into the folder /data/baci (http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37)<br/>
3. Open the py script "gravity_model.py" and run it.<br/>
  3.1 If you dont want to run the BACI data aggregation, which can take up to 12 hours depending on the number of O&Ds, I can provide you with an existing dataset. If    you are intersted in this pleace contact me (semfr@dtu.dk) and I am happy to provide you with this preestimation datasets. This should shorten the run-time per Scenario to about 20 minutes.<br/>
4. Analyse your results with the enclosed plotting scheme written in a RMarkdown file (SEACARTO_PLOTS.RMD), which enables to reproduce most figures from the article "Entering the antagonistic age of maritime trade and its impacts on energy demand". Unfortunately the geospatial data used to plot the maps in the article is not available open-source. However upon reasonable request I might be able to share them. Please contact me if you are interested (semfr@dtu.dk).
