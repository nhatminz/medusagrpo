def resolve_method(method):
    if method not in ('medusa','medusa_reflex'):
        raise ValueError('METHOD must be medusa or medusa_reflex')
    return method, method == 'medusa_reflex'
