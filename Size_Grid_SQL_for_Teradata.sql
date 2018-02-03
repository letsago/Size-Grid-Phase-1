/*      and CUST_INVTY_LOC_SIZE_WKLY.CUST_LOC_ID not in ('9300', '09312', '09313', '09315')
                    
    /*  91479 - JC Penney eCom
        91470 - JC Penney Stores */

    /* 91190 - Kohls everything */
    /* 91180 - Macys everything */

    /*
    Ecomm cust_loc_id 

    KOHLS DEPT STORE        00819
    KOHLS DEPT STORE        00873
    KOHLS DEPT STORE        00809
    KOHLS DEPT STORE        00829
    KOHLS DEPT STORE        00839

    MACY'S.com     0129 
    */

    /*  91479 - JC Penney eCom
    91470 - JC Penney Stores */

    /* 91190 - Kohls everything */
    /* 91180 - Macys everything */
    /* 91450 - Sears everything */

    /*
    Ecomm cust_loc_id 

    KOHLS DEPT STORE        00819
    KOHLS DEPT STORE        00873
    KOHLS DEPT STORE        00809
    KOHLS DEPT STORE        00829
    KOHLS DEPT STORE        00839

    MACY'S.com     0129 


    SEARS   9300 
    SEARS   09312
    SEARS   09313
    SEARS   09315 */

create volatile table TEMP_SALES_ONLY as
(
    select
        row_number() over (order by PROD_CD_9) as ID,
        CUSTOMER.CUST_MAIL_ADDR_NAME_1,
        CALENDAR_SEASON.FISC_YR_KEY,
        CALENDAR_SEASON.SEASON,
        CONSUMER_GRP.CONSM_GRP_LONG_DESC,
        PRODUCT.PROD_CD_9,
        SIZE.SIZE_DIM_1_DESC,
        SIZE.SIZE_DIM_2_DESC,
        sum(CUST_INVTY_LOC_SIZE_WKLY.SALES_UNITS) as SALES_QTY

    from
    /* Foundation Tables */
        EDWARD.CUST_INVTY_LOC_SIZE_WKLY

    /* Size Tables */
        left join EDWARD.SIZE on
            (CUST_INVTY_LOC_SIZE_WKLY.SIZE_DIM_1_CD = SIZE.SIZE_DIM_1_CD and CUST_INVTY_LOC_SIZE_WKLY.SIZE_DIM_2_CD = SIZE.SIZE_DIM_2_CD)

    /* Product Information */
        left join EDWARD.PRODUCT on
            (CUST_INVTY_LOC_SIZE_WKLY.PROD_KEY = PRODUCT.PROD_KEY)
        inner join EDWARD.PROD_SAP_MORE on
            (PRODUCT.PROD_KEY = PROD_SAP_MORE.PROD_KEY)

    /* Consumer Information */
        inner join EDWARD.CONSUMER_GRP_EXT on
            (PRODUCT.CONSM_GRP_EXT_ID = CONSUMER_GRP_EXT.CONSM_GRP_EXT_ID)
        inner join EDWARD.CONSUMER_GRP on
            (CONSUMER_GRP_EXT.CONSM_GRP_ID = CONSUMER_GRP.CONSM_GRP_ID)

    /* Customer Information */
        left join EDWARD.CUSTOMER on 
            (CUST_INVTY_LOC_SIZE_WKLY.CUST_ID = CUSTOMER.CUST_ID)

    /* Calendar Information */
        left join 
            (select CAL_DAY_KEY, FISC_YR_KEY, case when FISC_QTR_NUM <= 2 then 'SPRING' else 'FALL' end as "Season"
            from CALENDAR_DAY where FISC_YR_KEY = 2017) as CALENDAR_SEASON on                            
            (CUST_INVTY_LOC_SIZE_WKLY.CUST_INVTY_PER_END_DT = CALENDAR_SEASON.CAL_DAY_KEY)

    where
        CUST_INVTY_LOC_SIZE_WKLY.CUST_ID = '91470'
        and CALENDAR_SEASON.FISC_YR_KEY = 2017
        and CALENDAR_SEASON.SEASON like '%Fall%'
        and PRODUCT.PROD_CAT_CD like '%Bottoms%'
        and PRODUCT.PROD_SUBCAT_CD like '%Long%'
        and PROD_SAP_MORE.STYLE_NAME like '511 SLIM%'
        and CONSUMER_GRP.CONSM_GRP_ID = '001'
--        and CUST_INVTY_LOC_SIZE_WKLY.CUST_LOC_ID not in ('00819', '00873', '00809', '00829', '00839')
--        and (CUST_INVTY_LOC_SIZE_WKLY.END_INVTY_UNITS > 0 or (CUST_INVTY_LOC_SIZE_WKLY.END_INVTY_UNITS = 0 and CUST_INVTY_LOC_SIZE_WKLY.SALES_UNITS > 0))

    group by 2, 3, 4, 5, 6, 7, 8
    having SALES_QTY >= 0
) with data primary index (ID) on commit preserve rows;



create volatile table TEMP_SHIPMENTS as
(
    select
        row_number() over (order by PROD_CD_9) as ID,
        CUSTOMER.CUST_MAIL_ADDR_NAME_1,
        CALENDAR_SEASON.FISC_YR_KEY,
        CALENDAR_SEASON.SEASON,
        CONSUMER_GRP.CONSM_GRP_LONG_DESC,
        PRODUCT.PROD_CD_9,
        SIZE.SIZE_DIM_1_DESC,
        SIZE.SIZE_DIM_2_DESC,
        sum(ORD_ITEM_SIZE.ORD_SIZE_SHIP_QTY) as SHIPPED_UNITS

    from
/* Foundation Tables */
        EDWARD.ORD_ITEM
        inner join EDWARD.ORD_ITEM_SIZE on
            (ORD_ITEM.ORD_SRCE_CD = ORD_ITEM_SIZE.ORD_SRCE_CD and ORD_ITEM.ORD_CNTL_NUM = ORD_ITEM_SIZE.ORD_CNTL_NUM and ORD_ITEM.ORD_ENTRY_DT = ORD_ITEM_SIZE.ORD_ENTRY_DT and ORD_ITEM.POSNR_SLS_DOC_ITEM_NUM = ORD_ITEM_SIZE.POSNR_SLS_DOC_ITEM_NUM)
        inner join EDWARD.ORDER_HEADER on
            (ORD_ITEM.ORD_SRCE_CD = ORDER_HEADER.ORD_SRCE_CD and ORD_ITEM.ORD_CNTL_NUM = ORDER_HEADER.ORD_CNTL_NUM and ORD_ITEM.ORD_ENTRY_DT = ORDER_HEADER.ORD_ENTRY_DT)

/* Size Tables */
        left join EDWARD.SIZE on
            (ORD_ITEM_SIZE.SIZE_DIM_1_CD = SIZE.SIZE_DIM_1_CD and ORD_ITEM_SIZE.SIZE_DIM_2_CD = SIZE.SIZE_DIM_2_CD)

/* Product Information */
        left join EDWARD.PRODUCT on
            (ORD_ITEM.PROD_KEY = PRODUCT.PROD_KEY)
        inner join EDWARD.PROD_SAP_MORE on
            (PRODUCT.PROD_KEY = PROD_SAP_MORE.PROD_KEY)

/* Consumer Information */
        inner join EDWARD.CONSUMER_GRP_EXT on
            (PRODUCT.CONSM_GRP_EXT_ID = CONSUMER_GRP_EXT.CONSM_GRP_EXT_ID)
        inner join EDWARD.CONSUMER_GRP on
            (CONSUMER_GRP_EXT.CONSM_GRP_ID = CONSUMER_GRP.CONSM_GRP_ID)

/* Customer Information */
        left join EDWARD.CUSTOMER on 
            (ORDER_HEADER.CUST_ID = CUSTOMER.CUST_ID)

/* Calendar Information */
        left join 
            (select CAL_DAY_KEY, FISC_YR_KEY, case when FISC_QTR_NUM <= 2 then 'SPRING' else 'FALL' end as "Season"
            from CALENDAR_DAY) as CALENDAR_SEASON on                            
            (ORDER_HEADER.ORD_ROG_DT = CALENDAR_SEASON.CAL_DAY_KEY)

    where
        ORDER_HEADER.SHIP_FROM_LOC_ID in (2259332,2420476,9570815)
        and ORDER_HEADER.CUST_ID = '91470'
        and CALENDAR_SEASON.FISC_YR_KEY = 2017
        and CALENDAR_SEASON.SEASON like '%Fall%'
        and PRODUCT.PROD_CAT_CD like '%Bottoms%'
        and PRODUCT.PROD_SUBCAT_CD like '%Long%'
        and PROD_SAP_MORE.STYLE_NAME like '511 SLIM%'
        and CONSUMER_GRP.CONSM_GRP_ID = '001'
        and ORD_ITEM_SIZE.ORD_SIZE_SHIP_QTY > 0

    group by 2, 3, 4, 5, 6, 7, 8
) with data primary index (ID) on commit preserve rows;



/* Used to extract PC9 shipment data */
select
    TEMP_SALES_ONLY.CUST_MAIL_ADDR_NAME_1 as Planning_Group,
    TEMP_SALES_ONLY.FISC_YR_KEY as Fiscal_Year,
    TEMP_SALES_ONLY.SEASON,
    TEMP_SALES_ONLY.CONSM_GRP_LONG_DESC as Consumer_Group,
    TEMP_SALES_ONLY.PROD_CD_9 as PC9,
    sum(TEMP_SHIPMENTS.SHIPPED_UNITS) as PC9_Shipped_Qty
from 
    TEMP_SALES_ONLY right outer join TEMP_SHIPMENTS on 
        (TEMP_SHIPMENTS.CUST_MAIL_ADDR_NAME_1 = TEMP_SALES_ONLY.CUST_MAIL_ADDR_NAME_1 and TEMP_SHIPMENTS.PROD_CD_9 = TEMP_SALES_ONLY.PROD_CD_9 and TEMP_SHIPMENTS.SIZE_DIM_1_DESC = TEMP_SALES_ONLY.SIZE_DIM_1_DESC and TEMP_SHIPMENTS.SIZE_DIM_2_DESC = TEMP_SALES_ONLY.SIZE_DIM_2_DESC)
where 
    TEMP_SALES_ONLY.PROD_CD_9 is not null
group by 1, 2, 3, 4, 5



/* Used to extract PC9 size shipment data */
select
    TEMP_SALES_ONLY.PROD_CD_9 as PC9,
    TEMP_SALES_ONLY.SIZE_DIM_1_DESC,
    TEMP_SALES_ONLY.SIZE_DIM_2_DESC,
    sum(TEMP_SHIPMENTS.SHIPPED_UNITS) as Size_Shipped_Qty
from 
    TEMP_SALES_ONLY right outer join TEMP_SHIPMENTS on 
        (TEMP_SHIPMENTS.CUST_MAIL_ADDR_NAME_1 = TEMP_SALES_ONLY.CUST_MAIL_ADDR_NAME_1 and TEMP_SHIPMENTS.PROD_CD_9 = TEMP_SALES_ONLY.PROD_CD_9 and TEMP_SHIPMENTS.SIZE_DIM_1_DESC = TEMP_SALES_ONLY.SIZE_DIM_1_DESC and TEMP_SHIPMENTS.SIZE_DIM_2_DESC = TEMP_SALES_ONLY.SIZE_DIM_2_DESC)
where 
    TEMP_SALES_ONLY.PROD_CD_9 is not null
group by 1, 2, 3