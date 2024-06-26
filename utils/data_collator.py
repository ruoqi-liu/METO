from typing import Any, Dict, List, Optional, Tuple
import torch


class MyDataCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        outcome_labels = []
        treatment_labels = []
        token_type_ids = []
        visit_time_ids = []
        physical_time_ids = []
        treatment_drug_ids = []
        treatment_order_ids = []
        for b in batch:
            input_ids.append(b['input_ids'])
            attention_mask.append(b['attention_mask'])
            outcome_labels.append(b['label'])
            treatment_labels.append(b['treatment_label'])
            token_type_ids.append(b['token_type_ids'])
            visit_time_ids.append(b['visit_time_ids'])
            physical_time_ids.append(b['physical_time_ids'])
            treatment_drug_ids.append(b['treatment_drug_ids'])
            treatment_order_ids.append(b['treatment_order_ids'])

        input_ids = torch.tensor(input_ids,dtype=torch.long)
        attention_mask = torch.tensor(attention_mask,dtype=torch.long)
        outcome_labels = torch.tensor(outcome_labels)
        treatment_labels = torch.tensor(treatment_labels)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.long)
        visit_time_ids = torch.tensor(visit_time_ids, dtype=torch.long)
        physical_time_ids = torch.tensor(physical_time_ids, dtype=torch.long)
        treatment_drug_ids = torch.tensor(treatment_drug_ids, dtype=torch.long)
        treatment_order_ids = torch.tensor(treatment_order_ids, dtype=torch.long)
        batch = {"input_ids": input_ids,
                 "attention_mask": attention_mask,
                 "token_type_ids": token_type_ids,
                 "visit_time_ids": visit_time_ids,
                 "physical_time_ids": physical_time_ids,
                 "treatment_drug_ids": treatment_drug_ids,
                 "treatment_order_ids": treatment_order_ids,
                 "labels": outcome_labels,
                 "treatment_labels": treatment_labels}

        return batch