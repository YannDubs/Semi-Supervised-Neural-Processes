import argparse


def set_attributes(obj, **attributes):
    """Add attributes to a n object."""
    for k, v in attributes.items():
        setattr(obj, k, v)
    return obj


class FormatterNoDuplicate(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter overriding `argparse.ArgumentDefaultsHelpFormatter` to show
    `-e, --epoch EPOCH` instead of `-e EPOCH, --epoch EPOCH`

    Note
    ----
    - code modified from cPython: https://github.com/python/cpython/blob/master/Lib/argparse.py
    """

    def _format_action_invocation(self, action):
        # no args given
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)
            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # don't store the DEFAULT
                    parts.append('%s' % (option_string))
                # store DEFAULT for the last one
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)
