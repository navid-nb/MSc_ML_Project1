-- IBES actuals (alias measure -> act_measure)
SELECT
  ticker,
  anndats,
  anntims,
  pends,
  measure AS act_measure,
  pdicity
FROM ibes.actu_epsus
WHERE anndats < %(end)s::date;
