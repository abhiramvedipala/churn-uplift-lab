# Who should get the retention offer?

**To:** Retention manager · **Re:** targeting the discount offer · **Length:** one page

## The short answer

Send the offer to the customers it will actually change, not to the customers most
likely to leave. If the uplift model picks who gets it, offering it to 20% of our
customers (about 1,400 people) keeps an expected **354 customers** who would otherwise
have left. Offering it to the 1,400 highest-risk customers keeps **174**. Offering it
at random keeps **102**.

| If we send offers to... | Uplift model picks | Highest churn risk picks | Random |
|---|---|---|---|
| 10% of customers (~700) | 219 kept | 60 kept | 49 kept |
| 20% of customers (~1,400) | 354 kept | 174 kept | 102 kept |
| 30% of customers (~2,100) | 414 kept | 289 kept | 146 kept |
| Everyone (7,043) | 495 kept | 495 kept | 495 kept |

## Why "most likely to leave" is the wrong list

Customers fall into four groups, and only one of them is worth paying for:

- **Persuadables:** month-to-month customers, especially those on higher bills or
  without tech support. The discount changes their mind. *Target these.*
- **Lost causes:** fiber customers paying by electronic check. They are the most likely
  to leave, but they leave over the service, so a discount does not keep them. A
  risk-based list is full of them, which is why it keeps only half as many customers.
- **Sure things:** customers who would stay anyway. The offer costs money and changes
  nothing.
- **Sleeping dogs:** two-year contract customers. The offer reminds them to shop around
  and slightly *raises* their churn. *Never send to these.*

## Recommendation

1. Rank customers with the uplift model and send offers to the top **20-25%**. That is
   where the gain flattens; past 30%, each extra offer keeps few extra customers.
2. Exclude two-year contract customers from every campaign.
3. Keep using the churn-risk score to forecast how many customers we will lose, not to
   pick who gets the offer.

## Caveats (please read)

- **The offer was simulated.** We have no past retention campaign, so we built a
  realistic one with the group effects above to prove out the method. The "kept"
  counts show how well each targeting method recovers that effect; they are not a
  forecast for a real campaign.
- **Before rollout, run a real test.** Hold out a random half of the selected customers.
  To detect churn falling from 26.5% to 23.5% we need **3,268 customers per group**.
  Fix the test length in advance and do not stop early when results look good:
  checking 10 times along the way raises the false-alarm rate from 5% to **19.4%**.
- **The churn model is weaker on brand-new customers.** On customers in their first
  year it still ranks well (AUC-ROC 0.76) but under-predicts how many leave. Treat its
  probabilities as a ranking, not exact odds.

*All numbers come from `RESULTS.md`, produced by `scripts/check.sh`.*
