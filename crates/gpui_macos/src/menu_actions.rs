// Native menu tags are monotonic so callbacks from a replaced menu cannot dispatch a
// different action. Only the current main/Dock menu owns actions; tab churn must not
// grow a process-lifetime action history.
pub(crate) struct MenuActionSet<T> {
    first_tag: usize,
    actions: Vec<T>,
}

impl<T> MenuActionSet<T> {
    pub(crate) fn new(first_tag: usize) -> Self {
        Self {
            first_tag,
            actions: Vec::new(),
        }
    }

    pub(crate) fn next_tag(&self) -> usize {
        self.first_tag
            .checked_add(self.actions.len())
            .expect("native menu tag overflow")
    }

    pub(crate) fn push(&mut self, action: T) -> isize {
        let tag = isize::try_from(self.next_tag()).expect("native menu tag overflow");
        self.actions.push(action);
        tag
    }

    pub(crate) fn get(&self, tag: usize) -> Option<&T> {
        self.actions.get(tag.checked_sub(self.first_tag)?)
    }
}

impl<T> Default for MenuActionSet<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod menu_action_tests {
    use super::MenuActionSet;
    use std::rc::Rc;

    #[test]
    fn replacing_menus_releases_actions_and_rejects_stale_tags() {
        let payload = Rc::new(());
        let mut main = MenuActionSet::new(0);
        let stale = main.push(payload.clone()) as usize;
        let mut dock = MenuActionSet::new(main.next_tag());
        let dock_tag = dock.push(payload.clone()) as usize;
        let mut next_tag = dock.next_tag();
        for _ in 0..1000 {
            let mut replacement = MenuActionSet::new(next_tag);
            let tag = replacement.push(payload.clone()) as usize;
            next_tag = replacement.next_tag();
            main = replacement;
            assert_eq!(
                Rc::strong_count(&payload),
                3,
                "only two live menus own actions"
            );
            assert!(main.get(stale).is_none());
            assert!(main.get(dock_tag).is_none());
            assert!(dock.get(tag).is_none());
            assert!(main.get(tag).is_some());
            assert!(dock.get(dock_tag).is_some());
        }
        dock = MenuActionSet::new(next_tag);
        assert!(dock.get(dock_tag).is_none());
        assert_eq!(Rc::strong_count(&payload), 2);
        drop(main);
        assert_eq!(Rc::strong_count(&payload), 1);
    }
}
