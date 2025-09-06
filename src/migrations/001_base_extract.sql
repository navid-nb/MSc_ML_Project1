SELECT permno, date, ret, vol, prc, shrout
FROM crsp.dsf
WHERE date BETWEEN %(start)s AND %(end)s;