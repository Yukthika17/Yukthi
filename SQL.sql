1.
SELECT ord_no, purch_amt, ord_date, salesman_id 
FROM orders 
WHERE salesman_id IN(
SELECT salesman_id 
FROM salesman
WHERE commission = (
SELECT MAX(commission) 
FROM salesman));

2.
SELECT ord_no, purch_amt, ord_date, salesman_id 
FROM orders 
WHERE salesman_id IN(SELECT salesman_id FROM salesman WHERE commision = (SELECT MAX(commision) FROM salesman));

