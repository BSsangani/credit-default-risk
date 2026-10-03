-- default_by_latest_repayment_status
SELECT CASE WHEN PAY_1 <= 0 THEN 'Paid on time / no balance'
            WHEN PAY_1 = 1 THEN '1 month late' WHEN PAY_1 = 2 THEN '2 months late'
            ELSE '3+ months late' END AS latest_status,
       COUNT(*) AS customers, ROUND(100.0*AVG(DEFAULTED),1) AS default_rate_pct
FROM cards GROUP BY 1 ORDER BY default_rate_pct;

-- default_by_credit_limit_band
WITH b AS (SELECT *, NTILE(5) OVER (ORDER BY LIMIT_BAL) AS band FROM cards)
SELECT band AS limit_quintile, MIN(LIMIT_BAL) AS min_limit, MAX(LIMIT_BAL) AS max_limit,
       COUNT(*) AS customers, ROUND(100.0*AVG(DEFAULTED),1) AS default_rate_pct
FROM b GROUP BY band ORDER BY band;

-- default_by_utilisation
SELECT CASE WHEN 1.0*BILL_AMT1/LIMIT_BAL < 0.3 THEN 'a) under 30%' WHEN 1.0*BILL_AMT1/LIMIT_BAL < 0.7 THEN 'b) 30-70%'
            WHEN 1.0*BILL_AMT1/LIMIT_BAL < 1 THEN 'c) 70-100%' ELSE 'd) over limit' END AS utilisation,
       COUNT(*) AS customers, ROUND(100.0*AVG(DEFAULTED),1) AS default_rate_pct
FROM cards GROUP BY 1 ORDER BY 1;