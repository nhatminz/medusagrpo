"""Keep complete response rows together through GRPO length bucketing."""
GRPO_ALIGNMENT_VERSION = 'response_rows_v2'


def append_response_group(data, messages, rewards, advantages, prompt_id):
    if not (len(messages) == len(rewards) == len(advantages)):
        raise ValueError('response group lengths differ')
    metadata = data.setdefault('response_metadata', [])
    if len(metadata) != len(data['messages']):
        raise ValueError('missing response identity in pending training data')
    data['messages'].extend(messages)
    data['rewards'].extend(rewards)
    data['std_rewards'].extend(advantages)
    metadata.extend(dict(prompt_id=prompt_id, response_id=i) for i in range(len(messages)))


def sort_training_rows(input_ids, attention_mask, loss_mask, data):
    """Apply one stable permutation to tokens, masks, rewards and identities."""
    size = len(input_ids)
    fields = ('messages', 'rewards', 'std_rewards', 'response_metadata')
    if any(len(values) != size for values in (attention_mask, loss_mask)):
        raise ValueError('token/mask row counts differ')
    if any(len(data[key]) != size for key in fields):
        raise ValueError('response/advantage row counts differ')
    if any(len(ids) != len(attn) or len(ids) != len(mask)
           for ids, attn, mask in zip(input_ids, attention_mask, loss_mask)):
        raise ValueError('response token/mask lengths differ')
    order = sorted(range(size), key=lambda i: len(input_ids[i]))
    packed = tuple([values[i] for i in order] for values in (input_ids, attention_mask, loss_mask))
    for key in fields:
        data[key] = [data[key][i] for i in order]
    return packed
