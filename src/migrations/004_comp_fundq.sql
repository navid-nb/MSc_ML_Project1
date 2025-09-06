-- Compustat quarterly fundamentals (alias to safe identifiers)
SELECT
  gvkey,
  datadate,
  atq   AS "at",
  ltq   AS "lt",
  saleq AS "sale",
  niq   AS "ni"
FROM comp.fundq
WHERE datadate >= %(start)s::date
  AND datadate <  %(end)s::date;
