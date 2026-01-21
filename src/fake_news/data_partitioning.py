from sklearn.model_selection import train_test_split

def split_data(x, y, test_size=0.15, val_size=0.15, random_state=16):

  x_temp, x_test, y_temp, y_test = train_test_split(
      x, y,
      test_size=test_size,
      stratify=y,
      random_state=random_state
  )

  val_ratio_adjusted = val_size / (1 - test_size)

  x_train, x_val, y_train, y_val = train_test_split(
      x_temp, y_temp,
      test_size=val_ratio_adjusted,
      stratify=y_temp,
      random_state=random_state
  )

  return x_train, x_val, x_test, y_train, y_val, y_test

# Preparing mixed dataset for testing 

def build_mixed_test_set(x_test, y_test, x_title_test, y_title_test, x_body_test, y_body_test, TITLE_DIM, BODY_DIM):
    zeros_for_body = np.zeros((x_title_test.shape[0], BODY_DIM))
    x_title_test_padded = np.hstack([x_title_test, zeros_for_body])

    zeros_for_title = np.zeros((x_body_test.shape[0], TITLE_DIM))
    x_body_test_padded = np.hstack([zeros_for_title, x_body_test])
    
    x = np.vstack([x_test, x_title_test_padded, x_body_test_padded])
    y = np.concatenate([y_test, y_title_test, y_body_test])

    mask = (
        [[1, 1]] * len(x_test) +    # full
        [[1, 0]] * len(x_title_test) +   # title only
        [[0, 1]] * len(x_body_test)      # body only
    )

    idx = np.arange(len(y))
    np.random.shuffle(idx)

    return x[idx], y[idx], np.array(mask)[idx]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def weighted_early_fusion(
    x_title,
    x_body,
    y_wf_full,
    TITLE_WEIGHT=1.0,
    BODY_WEIGHT=1.15,
    test_size=0.3,
    val_size=0.5,
    random_state=42
):

    (
        x_wf_title_train,
        x_wf_title_temp,
        x_wf_body_train,
        x_wf_body_temp,
        y_wf_train,
        y_wf_temp
    ) = train_test_split(
        x_title,
        x_body,
        y_wf_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_wf_full
    )
    
    (
        x_wf_title_val,
        x_wf_title_test,
        x_wf_body_val,
        x_wf_body_test,
        y_wf_val,
        y_wf_test
    ) = train_test_split(
        x_wf_title_temp,
        x_wf_body_temp,
        y_wf_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_wf_temp
    )

    title_scaler = StandardScaler()
    body_scaler  = StandardScaler()

    x_wf_title_train_scaled = title_scaler.fit_transform(x_wf_title_train)
    x_wf_body_train_scaled  = body_scaler.fit_transform(x_wf_body_train)

    x_wf_title_val_scaled   = title_scaler.transform(x_wf_title_val)
    x_wf_body_val_scaled    = body_scaler.transform(x_wf_body_val)

    x_wf_title_test_scaled  = title_scaler.transform(x_wf_title_test)
    x_wf_body_test_scaled   = body_scaler.transform(x_wf_body_test)

    x_wf_train = np.concatenate(
        [TITLE_WEIGHT * x_wf_title_train_scaled,
         BODY_WEIGHT  * x_wf_body_train_scaled],
        axis=1
    )

    x_wf_val = np.concatenate(
        [TITLE_WEIGHT * x_wf_title_val_scaled,
         BODY_WEIGHT  * x_wf_body_val_scaled],
        axis=1
    )

    x_wf_test = np.concatenate(
        [TITLE_WEIGHT * x_wf_title_test_scaled,
         BODY_WEIGHT  * x_wf_body_test_scaled],
        axis=1
    )

    return {
        "x_wf_train": x_wf_train,
        "x_wf_val": x_wf_val,
        "x_wf_test": x_wf_test,
        "y_wf_train": y_wf_train,
        "y_wf_val": y_wf_val,
        "y_wf_test": y_wf_test,
        "title_scaler": title_scaler,
        "body_scaler": body_scaler
    }
