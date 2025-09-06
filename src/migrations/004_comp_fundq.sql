SELECT gvkey, datadate, atq AS at, ltq AS lt, saleq AS sale, niq AS ni
FROM comp.fundq q
WHERE datadate <= %(end)s
  AND gvkey IN (
     SELECT DISTINCT gvkey FROM comp.secm s
     WHERE datadate <= %(end)s
  );